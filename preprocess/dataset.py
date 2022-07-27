import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

from preprocess.helpers import *

class CustomValidationCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for monitoring performance on a holdout set
    Also implements restoration of best weights if chosen via command line argument
    """
    def __init__(self, train_ratings, train_mask, val_ratings, val_mask, means, stds, restore_best_weights=True):
        super(CustomValidationCallback, self).__init__()
        self.train_ratings = train_ratings
        self.train_mask = train_mask
        self.inputs = tf.stack([self.train_ratings, self.train_mask], axis=-1)
        self.val_ratings = val_ratings
        self.val_mask = val_mask
        self.means = means
        self.stds = stds
        self.restore_best_weights = restore_best_weights
        self.rmse_vals = []
        self.best_score = 42
        self.best_epoch = -1
        self.parameter_folder = os.path.join(".", "parameters", time.ctime())
        os.mkdir(self.parameter_folder)
        self.parameter_file = os.path.join(self.parameter_folder, "params")

    def on_epoch_end(self, epoch, logs=None):
        predictions = tf.maximum(1.0, tf.minimum(5.0, self.model.predict(self.inputs, batch_size=1<<10)[...,0] * self.stds + self.means))
        rmse = tf.sqrt(tf.reduce_sum(self.val_mask * tf.square(self.val_ratings - predictions)) / tf.maximum(tf.reduce_sum(self.val_mask), 1e-5))
        self.rmse_vals.append(rmse)
        if self.restore_best_weights:
            if rmse <= self.best_score:
                self.best_score = rmse
                self.best_epoch = epoch
                self.model.save_weights(self.parameter_file)

    def on_train_end(self, logs=None):
        if self.restore_best_weights:
            print(f"Restoring weights from epoch {self.best_epoch} with validation score {self.best_score}.")
            self.model.load_weights(self.parameter_file)
   
    def get_val_rmse(self):
        return self.rmse_vals

class CollaborativeFilteringDataset:
    """
    Dataset class takes care of loading the data, z-transforms, re-transforms and holdout sets
    Also handles storing the predictions
    """
    def __init__(self, dataset_folder, apply_z_trafo=True, normalize_by_col=False, val_split=0.1, store_dense_predictions=False):
        self.apply_z_trafo = apply_z_trafo
        self.normalize_by_col = normalize_by_col
        self.store_dense_predictions = store_dense_predictions

        # Dataset format info
        train_filepath = os.path.join(dataset_folder, "data_train.csv")
        predict_filepath = os.path.join(dataset_folder, "sampleSubmission.csv")
        self.n_rows = 10000
        self.n_cols = 1000
        
        # read training set and create validation split
        self.prep_train = read_and_preprocess(train_filepath)
        N = self.prep_train.last_valid_index()+1
        sample = np.random.rand(N) < val_split
        self.prep_val = self.prep_train[sample].copy()
        self.prep_train = self.prep_train[~sample].copy()
        
        # read prediction locations
        self.prep_predict = read_and_preprocess(predict_filepath)
        
        # Extract indices of sparse matrix
        self.train_indices = self.prep_train[["matrix_row", "matrix_col"]].to_numpy()
        self.val_indices = self.prep_val[["matrix_row", "matrix_col"]].to_numpy()
        self.pred_indices = self.prep_predict[["matrix_row", "matrix_col"]].to_numpy()
        
        # Normalize over valid entries only
        self.norm_train, self.mean_train, self.std_train = normalize(self.prep_train, by_col=normalize_by_col)
        
        # Extract ratings
        self.train_values = self.norm_train["Normalized" if self.apply_z_trafo else "Prediction"].to_numpy().reshape(-1, 1).astype(np.float32)
        self.val_values = self.prep_val["Prediction"].to_numpy().reshape(-1, 1).astype(np.float32)


    def get_matrix_dims(self):
        return self.n_rows, self.n_cols
    
    def get_dataset(self):
        return self.train_indices, self.train_values
    
    def get_sparse_mask(self):
        return tf.sparse.reorder(tf.sparse.SparseTensor(self.train_indices, np.ones((self.train_values.shape[0],), dtype=np.float32), (self.n_rows, self.n_cols)))
    
    def get_sparse_matrix(self):
        return tf.sparse.reorder(tf.sparse.SparseTensor(self.train_indices, self.train_values.reshape(-1), (self.n_rows, self.n_cols)))
    
    def get_dense_mask(self):
        return tf.sparse.to_dense(tf.sparse.reorder(self.get_sparse_mask()))

    def get_dense_matrix(self):
        valid_entries = tf.sparse.to_dense(tf.sparse.reorder(self.get_sparse_matrix()))
        if self.apply_z_trafo:
            # 0 is already row/col mean
            return valid_entries
        # impute missing values with row/col mean
        mask = self.get_dense_mask()
        ax = 0 if self.normalize_by_col else 1
        means = tf.reduce_sum(valid_entries, axis=ax, keepdims=True) / tf.reduce_sum(mask, axis=ax, keepdims=True)
        return valid_entries + (1. - mask) * means
    
    def get_validation_callback(self, restore_best_weights=True):
        train_ratings = self.get_dense_matrix()
        train_mask = self.get_dense_mask()

        mean_indices = np.array(self.mean_train.index)
        if self.normalize_by_col:
            mean_indices = np.stack([np.zeros_like(mean_indices), mean_indices], axis=-1)
        else:
            mean_indices = np.stack([mean_indices, np.zeros_like(mean_indices)], axis=-1)
        mean_values = self.mean_train.to_numpy().astype(np.float32)
        sparse_shape = (1, self.n_cols) if self.normalize_by_col else (self.n_rows, 1)
        mean_tensor = tf.sparse.to_dense(tf.sparse.SparseTensor(mean_indices, mean_values, sparse_shape))

        std_indices = np.array(self.std_train.index)
        if self.normalize_by_col:
            std_indices = np.stack([np.zeros_like(std_indices), std_indices], axis=-1)
        else:
            std_indices = np.stack([std_indices, np.zeros_like(std_indices)], axis=-1)
        std_values = self.std_train.to_numpy().astype(np.float32)
        std_tensor = tf.sparse.to_dense(tf.sparse.SparseTensor(std_indices, std_values, sparse_shape))
        if not self.apply_z_trafo: std_tensor = tf.ones_like(std_tensor)

        val_ratings = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(self.val_indices, self.val_values.reshape(-1), (self.n_rows, self.n_cols))))
        val_mask = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(self.val_indices, np.ones((self.val_indices.shape[0], ), dtype=np.float32), (self.n_rows, self.n_cols))))

        return CustomValidationCallback(train_ratings, train_mask, val_ratings, val_mask, mean_tensor, std_tensor, restore_best_weights=restore_best_weights)

    def get_val_locations(self):
        return self.val_indices

    def compute_val_score(self, locations, predictions):
        predictions = predictions.reshape(-1)
        assert(locations.shape[0] == predictions.shape[0])
        n = locations.shape[0]
        out_vals = []
        for i in range(n):
            row, col = locations[i]
            val = predictions[i]
            if self.apply_z_trafo and self.normalize_by_col:
                val *= self.std_train[col]
                val += self.mean_train[col]
            elif self.apply_z_trafo and not self.normalize_by_col:
                val *= self.std_train[row]
                val += self.mean_train[row]
            val = min(5.0, max(1.0, val))
            out_vals.append(val)

        score = tf.sqrt(tf.reduce_mean(tf.square(self.val_values.reshape(-1) - np.array(out_vals))))
        print(f"RMSE score: {score}")
        return score
    
    def compute_val_score_from_dense(self, dense_predictions):
        locations = self.get_val_locations()
        predictions = tf.gather_nd(dense_predictions, locations)
        return self.compute_val_score(locations, np.array(predictions))

    def get_prediction_locations(self):
        return self.pred_indices
        
    def create_submission(self, locations, predictions):
        predictions = predictions.reshape(-1)
        assert(locations.shape[0] == predictions.shape[0])
        n = locations.shape[0]
        out_ids = []
        out_vals = []
        for i in range(n):
            row, col = locations[i]
            val = predictions[i]
            if self.apply_z_trafo and self.normalize_by_col:
                val *= self.std_train[col]
                val += self.mean_train[col]
            elif self.apply_z_trafo and not self.normalize_by_col:
                val *= self.std_train[row]
                val += self.mean_train[row]
            val = min(5.0, max(1.0, val))
            out_ids.append(f"r{row+1}_c{col+1}")
            out_vals.append(val)
        
        ids = pd.Series(out_ids, name="Id")
        vals = pd.Series(out_vals, name="Prediction")
        df = pd.DataFrame(
            {
            "Id":ids, 
            "Prediction":vals
            }
        )
        timestamp = time.ctime()
        df.to_csv(os.path.join("predictions", f"{timestamp}.csv"), float_format="%.8f", index=False)
            
    def create_submission_from_dense(self, dense_predictions):
        locations = self.get_prediction_locations()
        values = tf.gather_nd(dense_predictions, locations)
        self.create_submission(locations, np.array(values))
        if self.store_dense_predictions:
            timestamp = time.ctime()
            predictions = dense_predictions.numpy()
            if self.apply_z_trafo:
                means = self.mean_train.to_numpy()
                stds = self.std_train.to_numpy()
                ax = 0 if self.normalize_by_col else 1
                predictions *= np.expand_dims(stds, axis=ax)
                predictions += np.expand_dims(means, axis=ax)

            np.savez_compressed(os.path.join("predictions_dense", f"{timestamp}.npz"), predictions=predictions)