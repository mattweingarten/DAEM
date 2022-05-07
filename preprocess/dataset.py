import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

from preprocess.helpers import *

class CollaborativeFilteringDataset:
    def __init__(self, dataset_folder, normalize_by_col=False):
        self.normalize_by_col = normalize_by_col

        train_filepath = os.path.join(dataset_folder, "data_train.csv")
        predict_filepath = os.path.join(dataset_folder, "sampleSubmission.csv")

        self.prep_train = read_and_preprocess(train_filepath)
        self.prep_predict = read_and_preprocess(predict_filepath)

        self.norm_train, self.mean_train, self.std_train = normalize(self.prep_train, by_col=normalize_by_col)
        row_input = self.norm_train["matrix_row"].to_numpy()
        col_input = self.norm_train["matrix_col"].to_numpy()
        self.inputs = np.stack([row_input, col_input], axis=-1)
        self.targets = self.norm_train["Normalized"].to_numpy().reshape(-1, 1).astype(np.float32)

    def get_train_test_split(self, test_fraction=0.1):
        return train_test_split(self.inputs, self.targets, shuffle=True, test_size=test_fraction)
    
    def get_dataset(self):
        return self.inputs, self.targets
    

