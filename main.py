import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares, train_and_predict_mf_ensemble
from model.slim import train_and_predict_SLIM
from model.ncf import train_and_predict_ncf_model
from model.kl_div import train_and_predict_kl_div
from model.baseline import train_and_predict_baseline
from model.svd import train_and_predict_low_rank_approx

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", apply_z_trafo=True, normalize_by_col=True, test_mode=False)

train_and_predict_alternating_least_squares(cil_dataset, k=3, iters=10, use_sgd=False, zeta=.75)