import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time

from sklearn.model_selection import train_test_split

from preprocess.helpers import *

class CollaborativeFilteringDataset:
    def __init__(self, dataset_folder, normalized=True, normalize_by_col=False):
        self.normalized = normalized
        self.normalize_by_col = normalize_by_col

        train_filepath = os.path.join(dataset_folder, "data_train.csv")
        predict_filepath = os.path.join(dataset_folder, "sampleSubmission.csv")

        self.prep_train = read_and_preprocess(train_filepath)
        self.prep_predict = read_and_preprocess(predict_filepath)

        self.norm_train, self.mean_train, self.std_train = normalize(self.prep_train, by_col=normalize_by_col)
        row_input = self.norm_train["matrix_row"].to_numpy()
        col_input = self.norm_train["matrix_col"].to_numpy()
        self.n_rows = np.max(row_input)+1
        self.n_cols = np.max(col_input)+1
        print(self.n_rows, self.n_cols)
        self.indices = np.stack([row_input, col_input], axis=-1)
        self.values = self.norm_train["Normalized" if self.normalized else "Prediction"].to_numpy().reshape(-1, 1).astype(np.float32)

    def get_matrix_dims(self):
        return self.n_rows, self.n_cols

    def get_train_test_split(self, test_fraction=0.1):
        return train_test_split(self.indices, self.values, shuffle=True, test_size=test_fraction)
    
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
        return self.prep_predict[["matrix_row", "matrix_col"]].to_numpy()
    
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
            val = min(5, max(0, round(val)))
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
        df.to_csv(os.path.join("predictions", f"{timestamp}.csv"), index=False)




