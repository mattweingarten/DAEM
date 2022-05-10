import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares
from model.slim import train_and_predict_SLIM
from model.ncf import train_and_predict_ncf_model

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", normalized=True, normalize_by_col=True, test_mode=True)


train_and_predict_alternating_least_squares(cil_dataset, 3, 0.1, 20, use_sgd=True)
