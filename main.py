import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares
from model.slim import train_and_predict_SLIM
from model.ncf import train_and_predict_ncf_model

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", normalized=False)

train_and_predict_alternating_least_squares(cil_dataset, 16, 0.01, 1000, use_sgd=True)
train_and_predict_alternating_least_squares(cil_dataset, 16, 0.01, 10, use_sgd=False)

train_and_predict_SLIM(cil_dataset, 1000, 0.01, 0.01)

train_and_predict_ncf_model(cil_dataset, 16, 40, model_type="ncf")
train_and_predict_ncf_model(cil_dataset, 16, 40, model_type="gmf")