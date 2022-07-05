from tkinter import Widget
import tensorflow as tf
from model.svd import low_rank_approx

def create_denoising_autoencoder(input_dim, width, depth, dropout_rate, strategy="standard"):
    
    inputs = tf.keras.Input(shape=(input_dim, 2), dtype=tf.float32)

    ratings = inputs[...,0]
    valid = inputs[...,1]

    layer_sizes = [width * (1<<i) for i in range(depth, 0, -1)] + [width] + [width * (2<<i) for i in range(depth)]

    print(f"Autoencoder shape: {layer_sizes}")

    mlp =  tf.keras.Sequential(
        [tf.keras.layers.Dense(x, activation="relu") for x in layer_sizes] +\
        [tf.keras.layers.Dense(input_dim)]
        )

    dropout = tf.keras.layers.Dropout(dropout_rate)(valid) # scaled by 1 / (1 - dropout_rate) during training
    dropout_mask = tf.cast(tf.math.greater(dropout, 0), tf.float32) # removed scaling, all values are 0, or 1

    if strategy=="compute effective dropout":
        effective_dropout_rate = 1. - tf.reduce_sum(dropout_mask, axis=-1, keepdims=True) / input_dim
        mlp_inputs = ratings * dropout_mask / tf.maximum(1. - effective_dropout_rate, 1e-5)
        rec = mlp(mlp_inputs)
    elif strategy=="normalize after dropout":
        means = tf.reduce_sum(dropout_mask * ratings, axis=1, keepdims=True) / tf.maximum(tf.reduce_sum(dropout_mask, axis=1, keepdims=True), 0.1)
        stds = tf.maximum(
            tf.sqrt(tf.reduce_sum(dropout_mask * tf.square(ratings - means), axis=1, keepdims=True) / tf.maximum(tf.reduce_sum(dropout_mask, axis=1, keepdims=True), 0.1)),
            0.001
        )
        mlp_inputs = dropout_mask * (ratings - means) / stds
        rec = mlp(mlp_inputs) * stds + means
    elif strategy=="no scaling":
        mlp_inputs = ratings * dropout_mask
        rec = mlp(mlp_inputs)
    elif strategy=="standard":
        mlp_inputs = ratings * dropout
        rec = mlp(mlp_inputs)
    else: raise f"Incompatible strategy: {strategy}"

    outputs = tf.stack([rec, dropout_mask], axis=-1)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def predict_autoenc(x, input_dim, width, depth, epochs=200, dropout_rate=0.5, strategy="standard"):
    
    def loss_fn(target_mask, prediction_mask):
        target = target_mask[...,0]
        valid_mask = target_mask[...,1]
        prediction = prediction_mask[...,0]
        dropout_mask = prediction_mask[...,1]
        mask = valid_mask * (1. - dropout_mask)
        return tf.reduce_sum(mask * tf.square(target - prediction)) / tf.maximum(tf.reduce_sum(mask), 1e-5)

    autoenc = create_denoising_autoencoder(input_dim, width, depth, dropout_rate, strategy=strategy)

    autoenc.compile(
        optimizer="Adam",
        loss=loss_fn
    )

    autoenc.fit(
        x=x,
        y=x,
        epochs=epochs,
        batch_size=1<<10,
        shuffle=True,
        verbose=0
    )

    dense_predictions = autoenc.predict(x, batch_size=1<<10)[...,0]

    return dense_predictions

def train_and_predict_autoencoder(dataset, width, depth, n=1, epochs=200, dropout_rate=0.5, strategy="standard"):
    x = tf.stack([dataset.get_dense_matrix(), dataset.get_dense_mask()], axis=-1)
    n_samples, input_dim = dataset.get_matrix_dims()
    
    dense_predictions = tf.zeros(dataset.get_matrix_dims(), dtype=tf.float32)

    for i in range(n):
        dense_predictions += (1. / n) * predict_autoenc(x, input_dim, width, depth, epochs=epochs, dropout_rate=dropout_rate, strategy=strategy)
    
    return dataset.create_submission_from_dense(dense_predictions) # return value only relevant if dataset uses hold out set and returns estimated score
