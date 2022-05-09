import tensorflow as tf

def create_generalized_matrix_factorization_model(n_rows, n_cols, n_latent=3, lamb=0.05):
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

    row_latent = row_emb(row_idx)[:,0,:]
    col_latent = col_emb(col_idx)[:,0,:]

    row_b = row_bias(row_idx)[:,0,:]
    col_b = col_bias(col_idx)[:,0,:]

    interm = tf.concat([row_latent * col_latent, row_b, col_b], axis=1)
    
    generalized_inner_product_layer = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Constant(value=1.))

    outputs = generalized_inner_product_layer(interm)

    return tf.keras.Model(inputs, outputs)

def create_neural_collaborative_filtering_model(n_rows, n_cols, n_latent=16, lamb=0.05):
    row_emb = tf.keras.layers.Embedding(
        n_rows, n_latent,
        embeddings_initializer="glorot_uniform",
        embeddings_regularizer=tf.keras.regularizers.L2(l2=lamb)
    )
    col_emb = tf.keras.layers.Embedding(
        n_cols, n_latent,
        embeddings_initializer="glorot_uniform",
        embeddings_regularizer=tf.keras.regularizers.L2(l2=lamb)
    )
    inputs = tf.keras.Input(shape=(2,), dtype=tf.int64)
    row_idx = inputs[:, :1]
    col_idx = inputs[:, 1:]

    row_latent = row_emb(row_idx)[:,0,:]
    col_latent = col_emb(col_idx)[:,0,:]

    interm = tf.concat([row_latent, col_latent], axis=1)

    mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(2*n_latent, activation="relu"),
        tf.keras.layers.Dense(n_latent, activation="relu"),
        tf.keras.layers.Dense(n_latent//2, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    outputs = mlp(interm)

    return tf.keras.Model(inputs, outputs)

def train_and_predict_ncf_model(
    dataset, n_latent, epochs, model_type="ncf"
):
    n_rows, n_cols = dataset.get_matrix_dims()
    model = None
    if model_type=="ncf":
        model = create_neural_collaborative_filtering_model(n_rows, n_cols, n_latent=n_latent)
    elif model_type=="gmf":
        model = create_generalized_matrix_factorization_model(n_rows, n_cols, n_latent=n_latent)
    else: raise(f"Unknown model type: {model_type}")
    
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.MeanSquaredError()
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
