import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import math

from preprocess.helpers import *

class CollaborativeFilteringDataset:
    def __init__(self, dataset_folder, normalized=True, normalize_by_col=False, test_mode=False):
        self.normalized = normalized
        self.normalize_by_col = normalize_by_col
        self.test_mode = test_mode

        train_filepath = os.path.join(dataset_folder, "data_train.csv")
        predict_filepath = os.path.join(dataset_folder, "sampleSubmission.csv")

        self.prep_train = read_and_preprocess(train_filepath)
        self.prep_predict = read_and_preprocess(predict_filepath)

        self.norm_train, self.mean_train, self.std_train = normalize(self.prep_train, by_col=normalize_by_col)
        self.indices = self.norm_train[["matrix_row", "matrix_col"]].to_numpy()
        self.n_rows = np.max(self.indices[:, 0])+1
        self.n_cols = np.max(self.indices[:, 1])+1
        self.values = self.norm_train["Normalized" if self.normalized else "Prediction"].to_numpy().reshape(-1, 1).astype(np.float32)

        if test_mode:
            N = self.indices.shape[0]
            sample = np.random.rand(N) < 0.05
            self.pred_indices = self.indices[sample]
            self.indices = self.indices[~sample]
            self.pred_targets = self.norm_train["Prediction"].to_numpy().reshape(-1, 1).astype(np.float32)[sample]
            self.values = self.values[~sample]
        else:
            self.pred_indices = self.prep_predict[["matrix_row","matrix_col"]].to_numpy()
            self.pred_targets = None

    def get_matrix_dims(self):
        return self.n_rows, self.n_cols
    
    def get_dataset(self):
        return self.indices, self.values
    
    def get_sparse_matrix(self):
        return tf.sparse.SparseTensor(self.indices, self.values.reshape(-1), (self.n_rows, self.n_cols))

    def get_dense_matrix(self):
        return tf.sparse.to_dense(tf.sparse.reorder(self.get_sparse_matrix()))

    def get_slim_dataset(self):
        targets = tf.sparse.transpose(
            self.get_sparse_matrix(),
            perm=[1,0]
        ) # n_cols x n_rows
        inputs = tf.expand_dims(
            tf.range(0, self.n_cols, dtype=tf.int64),
            axis=-1
        ) # n_cols x 1

        return inputs, tf.sparse.to_dense(targets)
    
    def get_prediction_locations(self):
        return self.pred_indices
    
    def postprocess_and_save(self, locations, predictions):
        predictions = predictions.reshape(-1)
        assert(locations.shape[0] == predictions.shape[0])
        n = locations.shape[0]
        out_ids = []
        out_vals = []
        for i in range(n):
            row, col = locations[i]
            val = predictions[i]
            if self.normalized and self.normalize_by_col:
                val *= self.std_train[col]
                val += self.mean_train[col]
            elif self.normalized and not self.normalize_by_col:
                val *= self.std_train[row]
                val += self.mean_train[row]
            val = min(5, max(1, round(val)))
            out_ids.append(f"r{row+1}_c{col+1}")
            out_vals.append(val)
        if self.test_mode:
            vals = np.array(out_vals).reshape(-1, 1)
            rmse = math.sqrt(np.mean((vals - self.pred_targets)**2))
            print(f"RMSE score: {rmse}")
        else:
            ids = pd.Series(out_ids, name="Id")
            vals = pd.Series(out_vals, name="Prediction")
            df = pd.DataFrame(
                {
                "Id":ids, 
                "Prediction":vals
                }
            )
            timestamp = time.ctime()
            df.to_csv(os.path.join("predictions", f"{timestamp}.csv"), index=False)




