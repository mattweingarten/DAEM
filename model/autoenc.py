import tensorflow as tf

def create_denoising_autoencoder(input_dim, layer_sizes, dropout_rate, strategy="standard"):
    
    inputs = tf.keras.Input(shape=(input_dim, 2), dtype=tf.float32)

    ratings = inputs[...,0]
    valid = inputs[...,1]

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

def train_and_predict_autoencoder(dataset, axis, layer_sizes, epochs=200, dropout_rate=0.5, reduction_dims=[0,1], strategy="standard"):
    """
    axis==0: encode users rating vectors
    axis==1: encode items rating vectors
    """
    n_rows, n_cols = dataset.get_matrix_dims()

    assert(axis==0 or axis==1)
    input_dim = n_rows if axis==1 else n_cols
    n_samples = n_rows if axis==0 else n_cols
    samples = dataset.get_dense_matrix()
    mask_samples = dataset.get_dense_mask()
    if axis==1:
        samples = tf.transpose(samples, perm=[1,0])
        mask_samples = tf.transpose(mask_samples, perm=[1,0])
    

    x = tf.stack([samples, mask_samples], axis=-1)

    def loss_fn(target_mask, prediction_mask):
        target = target_mask[...,0]
        valid_mask = target_mask[...,1]
        prediction = prediction_mask[...,0]
        dropout_mask = prediction_mask[...,1]
        mask = valid_mask * (1. - dropout_mask)
        return tf.reduce_mean(
            tf.reduce_sum(mask * tf.square(target - prediction), axis=reduction_dims, keepdims=True) / (tf.reduce_sum(mask, axis=reduction_dims, keepdims=True) + 1e-5)
        )

    autoenc = create_denoising_autoencoder(input_dim, layer_sizes, dropout_rate, strategy=strategy)

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
    if axis==1:
        dense_predictions = tf.transpose(dense_predictions, perm=[1,0])

    dataset.create_submission_from_dense(dense_predictions)
