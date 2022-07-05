import tensorflow as tf

from model.als import init_latent_vectors

def kl_div_matrix_factorization(matrix, train_samples, k, beta, epochs):
    rows, cols = matrix.shape

    row_emb_mean = tf.keras.layers.Embedding(rows, k)
    row_emb_logvar = tf.keras.layers.Embedding(rows, k, embeddings_initializer=tf.keras.initializers.Constant(-5))

    col_emb_mean = tf.keras.layers.Embedding(cols, k)
    col_emb_logvar = tf.keras.layers.Embedding(cols, k, embeddings_initializer=tf.keras.initializers.Constant(-5))

    row_emb_mean(tf.zeros((1,1),dtype=tf.int32))
    col_emb_mean(tf.zeros((1,1),dtype=tf.int32))
    U, V = init_latent_vectors(matrix, k)
    row_emb_mean.set_weights([U.numpy()])
    col_emb_mean.set_weights([V.numpy()])

    data = tf.data.Dataset.from_tensor_slices(train_samples).shuffle(1<<14).batch(1<<12)
    
    def kl_loss(mean, logvar):
        return  tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1. + logvar - tf.square(mean) - tf.exp(logvar)),
                axis=1)
                )
    
    opt = tf.keras.optimizers.Adam()
    for i in range(epochs):
        trainable_weights = (row_emb_mean.trainable_weights + row_emb_logvar.trainable_weights) \
                            if i%2==0 else \
                            (col_emb_mean.trainable_weights + col_emb_logvar.trainable_weights)
        metric_pred_loss = 0
        for x, y in data:
            row_ids = x[:,:1]
            col_ids = x[:,1:]
            with tf.GradientTape() as tape:
                row_mean = row_emb_mean(row_ids)[:, 0, :]
                row_logvar = row_emb_logvar(row_ids)[:, 0, :]
                col_mean = col_emb_mean(col_ids)[:, 0, :]
                col_logvar = col_emb_logvar(col_ids)[:, 0, :]
                row_eps = tf.keras.backend.random_normal(shape=(1, k))
                col_eps = tf.keras.backend.random_normal(shape=(1, k))
                row_latent = row_mean + row_eps * tf.exp(row_logvar * 0.5)
                col_latent = col_mean + col_eps * tf.exp(col_logvar * 0.5)
                prediction = tf.reduce_sum(
                    row_latent * col_latent,
                    axis=1, keepdims=True
                )
                pred_loss = tf.reduce_mean(tf.square(prediction - y))
                row_kl = kl_loss(row_mean, row_logvar)
                col_kl = kl_loss(col_mean, col_logvar)
                total_loss = pred_loss + beta*row_kl + beta*col_kl
            grads = tape.gradient(total_loss, trainable_weights)
            opt.apply_gradients(zip(grads, trainable_weights))
            metric_pred_loss += pred_loss
        print(f"it. {i}, rec. loss: {metric_pred_loss/len(data)}")
        
    row_mean_vectors = row_emb_mean.get_weights()[0]
    col_mean_vectors = col_emb_mean.get_weights()[0]
    dense_predictions = row_mean_vectors @ tf.transpose(col_mean_vectors, perm=[1,0])

    return dense_predictions


def train_and_predict_kl_div(dataset, k=32, beta=0.05, epochs=10):

    dense_predictions = kl_div_matrix_factorization(dataset.get_dense_matrix(), dataset.get_dataset(), k, beta, epochs)

    return dataset.create_submission_from_dense(dense_predictions)


    