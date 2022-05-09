import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares
from model.slim import train_and_predict_SLIM
from model.ncf import train_and_predict_ncf_model

cil_dataset_col = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", normalized=True, normalize_by_col=True, test_mode=True)
cil_dataset_row = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", normalized=True, normalize_by_col=False, test_mode=True)
cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", normalized=False, normalize_by_col=False, test_mode=True)

train_and_predict_SLIM(cil_dataset_col, 1000, 0.1, 0.1)
train_and_predict_SLIM(cil_dataset_row, 1000, 0.1, 0.1)
train_and_predict_SLIM(cil_dataset, 1000, 0.1, 0.1)


#train_and_predict_alternating_least_squares(cil_dataset, 3, 0.1, 1000, use_sgd=True)
train_and_predict_alternating_least_squares(cil_dataset_col, 3, 0.1, 5, use_sgd=False)
train_and_predict_alternating_least_squares(cil_dataset_row, 3, 0.1, 5, use_sgd=False)
train_and_predict_alternating_least_squares(cil_dataset, 3, 0.1, 5, use_sgd=False)


train_and_predict_ncf_model(cil_dataset_col, 3, 40, model_type="ncf")
train_and_predict_ncf_model(cil_dataset_row, 3, 40, model_type="ncf")
train_and_predict_ncf_model(cil_dataset, 3, 40, model_type="ncf")

train_and_predict_ncf_model(cil_dataset_col, 3, 40, model_type="gmf")
train_and_predict_ncf_model(cil_dataset_row, 3, 40, model_type="gmf")
train_and_predict_ncf_model(cil_dataset, 3, 40, model_type="gmf")