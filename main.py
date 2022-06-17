import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares, train_and_predict_mf_ensemble
from model.slim import train_and_predict_SLIM
from model.ncf import train_and_predict_ncf_model
from model.kl_div import train_and_predict_kl_div
from model.baseline import train_and_predict_baseline

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", normalized=True, normalize_by_col=True, test_mode=True)

train_and_predict_alternating_least_squares(cil_dataset, k=2, lamb=0.1, use_sgd=True)
train_and_predict_alternating_least_squares(cil_dataset, k=2, lamb=1.0, use_sgd=True)
train_and_predict_alternating_least_squares(cil_dataset, k=2, lamb=10., use_sgd=True)