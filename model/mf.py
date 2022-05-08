import tensorflow as tf

def create_matrix_factorization_model(n_rows, n_cols, n_latent=128, lamb=0.05):
    initializer = tf.keras.initializers.RandomUniform(minval=-.1, maxval=.1)

    row_emb = tf.keras.layers.Embedding(
        n_rows, n_latent,
        embeddings_initializer="glorot_uniform",
        embeddings_regularizer=tf.keras.regularizers.L2(l2=lamb)
    )
    row_bias = tf.keras.layers.Embedding(
        n_rows, 1,
        embeddings_initializer="glorot_uniform",
        embeddings_regularizer=tf.keras.regularizers.L2(l2=lamb)
    )
    col_emb = tf.keras.layers.Embedding(
        n_cols, n_latent,
        embeddings_initializer="glorot_uniform",
        embeddings_regularizer=tf.keras.regularizers.L2(l2=lamb)
    )
    col_bias = tf.keras.layers.Embedding(
        n_cols, 1,
        embeddings_initializer="glorot_uniform",
        embeddings_regularizer=tf.keras.regularizers.L2(l2=lamb)
    )

    inputs = tf.keras.Input(shape=(2,), dtype=tf.int64)
    row_idx = inputs[:, :1]
    col_idx = inputs[:, 1:]

    row_latent = row_emb(row_idx)
    col_latent = col_emb(col_idx)

    row_b = row_bias(row_idx)[...,0]
    col_b = col_bias(col_idx)[...,0]

    inner_product = tf.reduce_sum(row_latent * col_latent, axis=-1, keepdims=False)

    bias_layer = tf.keras.layers.Dense(1)

    outputs = bias_layer(inner_product + row_b + col_b)

    return tf.keras.Model(inputs, outputs)

def train_and_predict_sgd_matrix_factorization(
    dataset, n_latent, epochs
):
    n_rows, n_cols = dataset.get_matrix_dims()
    model = create_matrix_factorization_model(n_rows, n_cols, n_latent=n_latent)
    model.compile(
        optimizer="Adam",
        loss="mean_squared_error"
    )

    inputs, targets = dataset.get_dataset()
    model.fit(
        x=inputs,
        y=targets,
        batch_size=1024,
        epochs=epochs,
        validation_split=0.01,
        shuffle=True
    )

    locations = dataset.get_prediction_locations()
    values = model.predict(locations, batch_size=1024)
    dataset.postprocess_and_save(locations, values)
