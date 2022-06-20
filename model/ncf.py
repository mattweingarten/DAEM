import tensorflow as tf
from model.als import init_latent_vectors

def create_generalized_matrix_factorization_model(n_rows, n_cols, n_latent, lamb=0.1):
    row_emb = tf.keras.layers.Embedding(
        n_rows, n_latent,
        embeddings_initializer="glorot_uniform",
        embeddings_regularizer=tf.keras.regularizers.L2(l2=lamb)
    )
    row_bias = tf.keras.layers.Embedding(
        n_rows, 1,
        embeddings_initializer="zero"
    )
    col_emb = tf.keras.layers.Embedding(
        n_cols, n_latent,
        embeddings_initializer="glorot_uniform",
        embeddings_regularizer=tf.keras.regularizers.L2(l2=lamb)
    )
    col_bias = tf.keras.layers.Embedding(
        n_cols, 1,
        embeddings_initializer="zero"
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

    return tf.keras.Model(inputs, outputs), [row_emb, row_bias], [col_emb, col_bias]

def create_neural_collaborative_filtering_model(n_rows, n_cols, n_latent, lamb=0.1):
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

    return tf.keras.Model(inputs, outputs), [row_emb], [col_emb]

def create_neural_bi_form(n_rows, n_cols, n_latent, lamb=0.1):
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

    bilinear_forms = tf.expand_dims(
        tf.expand_dims(row_latent, axis=1) * tf.expand_dims(col_latent, axis=2),
        axis=3
    )

    mlp = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(n_latent*n_latent, activation="relu"),
        tf.keras.layers.Dense(n_latent*n_latent, activation="relu"),
        tf.keras.layers.Dense(n_latent, activation="relu"),
        tf.keras.layers.Dense(n_latent, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    outputs = mlp(bilinear_forms)

    return tf.keras.Model(inputs, outputs), [row_emb], [col_emb]



def train_and_predict_ncf_model(
    dataset, n_latent=16, epochs=20, model_type="nbf"
):
    n_rows, n_cols = dataset.get_matrix_dims()
    model = None
    if model_type=="ncf":
        model, row_latent_weights, col_latent_weights = create_neural_collaborative_filtering_model(n_rows, n_cols, n_latent=n_latent)
    elif model_type=="gmf":
        model, row_latent_weights, col_latent_weights = create_generalized_matrix_factorization_model(n_rows, n_cols, n_latent=n_latent)
    elif model_type=="nbf":
        model, row_latent_weights, col_latent_weights = create_neural_bi_form(n_rows, n_cols, n_latent=n_latent)
    else: raise(f"Unknown model type: {model_type}")

    Users, Items = init_latent_vectors(dataset.get_dense_matrix(),n_latent)
    row_latent_weights[0].set_weights([Users.numpy()])
    col_latent_weights[0].set_weights([Items.numpy()])
    
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.MeanSquaredError()
    )

    inputs, targets = dataset.get_dataset()

    for i in range(epochs):
        for l in row_latent_weights: l.trainable = True
        for l in col_latent_weights: l.trainable = False

        h1 = model.fit(
            x=inputs,
            y=targets,
            batch_size=1<<16,
            epochs=20,
            verbose=0,
            shuffle=True
        )

        print(f"it {i}, User step. Loss {h1.history['loss'][-1]}")

        for l in row_latent_weights: l.trainable = False
        for l in col_latent_weights: l.trainable = True

        h2 = model.fit(
            x=inputs,
            y=targets,
            batch_size=1<<16,
            epochs=20,
            verbose=0,
            shuffle=True
        )

        print(f"it {i}, Item step. Loss {h2.history['loss'][-1]}")

    locations = dataset.get_prediction_locations()
    values = model.predict(locations, batch_size=1024)
    dataset.create_submission(locations, values)
