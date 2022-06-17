import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares, train_and_predict_mf_ensemble
from model.slim import train_and_predict_SLIM
from model.ncf import train_and_predict_ncf_model
from model.kl_div import train_and_predict_kl_div

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", normalized=True, normalize_by_col=True, test_mode=True)


train_and_predict_alternating_least_squares(cil_dataset, use_sgd=True, iters=20)
