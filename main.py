import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares
from model.mf import train_and_predict_sgd_matrix_factorization

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022")

train_and_predict_sgd_matrix_factorization(cil_dataset, 3, 10)