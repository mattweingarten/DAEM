import tensorflow as tf


def compute_slim_matrix(
    A, mask, iters, l1, l2
):
    n_rows, n_cols = A.shape
    
    W = tf.Variable(
        tf.random.uniform((n_cols, n_cols), minval=0, maxval=2./n_cols)
    )

    def metric_fn(matrix):
        return tf.reduce_sum(mask*(A - A@matrix)**2) / tf.reduce_sum(mask)

    def loss_fn(matrix):
        return tf.reduce_sum(mask*(A - A@matrix)**2) + l2/2 *tf.reduce_sum(matrix**2) + l1 * tf.reduce_sum(tf.math.abs(matrix))

    def project(matrix):
        return matrix * (1. - tf.eye(n_cols)) * tf.cast(tf.math.greater_equal(matrix, 0), tf.float32)

    W.assign(project(W))

    opt = tf.keras.optimizers.Adam()

    print(f"Starting loss: {metric_fn(W)}")

    for i in range(iters):
        with tf.GradientTape() as tape:
            loss = loss_fn(W)
        grads = tape.gradient(loss, [W])
        opt.apply_gradients(zip(grads, [W]))
        W.assign(project(W))
        print(f"Loss after it. {i}: {metric_fn(W)}")
    
    return W

def train_and_predict_SLIM(
    dataset, iters=5000, l1=0.1, l2=0.0
):
    A = dataset.get_dense_matrix()
    Omega = dataset.get_dense_mask()
    W = compute_slim_matrix(A, Omega, iters, l1, l2)
    A_tilde = A @ W

    return A_tilde
