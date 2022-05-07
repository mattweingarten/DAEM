import numpy as np
import tensorflow as tf

from preprocess.dataset import CollaborativeFilteringDataset
from model.mf import create_matrix_factorization_model

epochs = 200
batch_size = 4096
test_fraction = 0.01

cil_dataset = CollaborativeFilteringDataset("~/datasets/cil-collaborative-filtering-2022")
x, y = cil_dataset.get_dataset()

model = create_matrix_factorization_model(10000, 1000)
model.compile(
    optimizer="Adam",
    loss="mean_squared_error"
)
model.fit(
    x=x,
    y=y,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.01
)