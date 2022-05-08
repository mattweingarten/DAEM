import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares

epochs = 400
batch_size = 4096
test_fraction = 0.01

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022")

train_and_predict_alternating_least_squares(cil_dataset, 3, 0.1, 1)