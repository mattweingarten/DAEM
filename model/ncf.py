from pickletools import optimize
import tensorflow as tf

def create_gmf_model(n_rows, n_cols, n_latent):
    row_emb = tf.keras.layers.Embedding(
        n_rows, n_latent
    )
    col_emb = tf.keras.layers.Embedding(
        n_cols, n_latent
    )

    inputs = tf.keras.Input(shape=(2,), dtype=tf.int64)
    row_idx = inputs[:, :1]
    col_idx = inputs[:, 1:]

    row_latent = row_emb(row_idx)[:,0,:]
    col_latent = col_emb(col_idx)[:,0,:]
   
    outputs = row_latent * col_latent

    return tf.keras.Model(inputs, outputs)

def create_mlp_model(n_rows, n_cols, predictive_factors, layers=4):
    n_latent = predictive_factors<<(layers-1)
    row_emb = tf.keras.layers.Embedding(
        n_rows, n_latent
    )
    col_emb = tf.keras.layers.Embedding(
        n_cols, n_latent
    )
    inputs = tf.keras.Input(shape=(2,), dtype=tf.int64)
    row_idx = inputs[:, :1]
    col_idx = inputs[:, 1:]

    row_latent = row_emb(row_idx)[:,0,:]
    col_latent = col_emb(col_idx)[:,0,:]

    interm = tf.concat([row_latent, col_latent], axis=1)

    mlp = tf.keras.Sequential([tf.keras.layers.Dense(predictive_factors<<(i-1), activation="relu") for i in range(layers, 0, -1)])

    outputs = mlp(interm)

    return tf.keras.Model(inputs, outputs)


def train_and_predict_ncf_model(
    dataset, n_latent=16, layers=4, epochs=20, model_type="ncf"
):
    n_rows, n_cols = dataset.get_matrix_dims()
    inputs, targets = dataset.get_dataset()
    model = None
    if model_type=="mlp":
        model = tf.keras.Sequential([
            create_mlp_model(n_rows, n_cols, n_latent, layers=layers),
            tf.keras.layers.Dense(1)
        ])
    elif model_type=="gmf":
        model = tf.keras.Sequential([
            create_gmf_model(n_rows, n_cols, n_latent),
            tf.keras.layers.Dense(1)
        ])
    elif model_type=="ncf":
        mlp_model = create_mlp_model(n_rows, n_cols, n_latent, layers=layers)
        pretrain_mlp_model = tf.keras.Sequential([
            mlp_model,
            tf.keras.layers.Dense(1)
        ])
        gmf_model = create_gmf_model(n_rows, n_cols, n_latent)
        pretrain_gmf_model = tf.keras.Sequential([
            gmf_model,
            tf.keras.layers.Dense(1)
        ])

        pretrain_mlp_model.compile(
            optimizer="Adam",
            loss=tf.keras.losses.MeanSquaredError()
        )
        pretrain_mlp_model.fit(
            x=inputs,
            y=targets,
            batch_size=1<<10,
            epochs=epochs,
            verbose=0,
            shuffle=True
        )

        pretrain_gmf_model.compile(
            optimizer="Adam",
            loss=tf.keras.losses.MeanSquaredError()
        )
        pretrain_gmf_model.fit(
            x=inputs,
            y=targets,
            batch_size=1<<10,
            epochs=epochs,
            verbose=0,
            shuffle=True
        )

        model_inputs = tf.keras.Input(shape=(2,), dtype=tf.int64)
        mlp_factors = mlp_model(model_inputs)
        gmf_factors = gmf_model(model_inputs)
        alpha=0.5
        combined_factors = tf.concat([alpha * mlp_factors, (1.-alpha) * gmf_factors], axis=-1)
        model_outputs = tf.keras.layers.Dense(1)(combined_factors)
        model = tf.keras.Model(model_inputs, model_outputs)
    else: raise(f"Unknown model type: {model_type}")
    
    model.compile(
        optimizer="SGD" if model_type=="ncf" else "Adam",
        loss=tf.keras.losses.MeanSquaredError()
    )


    model.fit(
        x=inputs,
        y=targets,
        batch_size=1<<10,
        epochs=epochs,
        verbose=0,
        shuffle=True
    )

    return model
