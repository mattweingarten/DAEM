import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.als import train_and_predict_alternating_least_squares, train_and_predict_mf_ensemble, train_and_predict_modular_als
from model.slim import train_and_predict_SLIM
from model.ncf import train_and_predict_ncf_model
from model.kl_div import train_and_predict_kl_div
from model.baseline import train_and_predict_baseline
from model.svd import train_and_predict_low_rank_approx
from model.autoenc import train_and_predict_autoencoder

from model.weighting_schemes import *

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022", apply_z_trafo=True, normalize_by_col=True, test_mode=True)

# Second best approach:
#Omega = cil_dataset.get_dense_mask()
#user_weights = inverse_frequency_weights(Omega, 1)
#item_weights = inverse_frequency_weights(Omega, 0)
#train_and_predict_modular_als(cil_dataset, lambda A, A_tilde: tf.square(A - A_tilde), user_weights=user_weights, item_weights=item_weights)

# Current best approach
train_and_predict_autoencoder(cil_dataset, width=8, depth=3, n=1, epochs=200, dropout_rate=0.5, strategy="standard")