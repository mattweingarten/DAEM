import tensorflow as tf
import numpy as np

def init_latent_vectors(mat, k):
    sigma, Ufull, Vfull = tf.linalg.svd(mat)

    U = Ufull[:, :k] * tf.expand_dims(tf.math.sqrt(sigma[:k]), axis=0)
    V = Vfull[:, :k] * tf.expand_dims(tf.math.sqrt(sigma[:k]), axis=0)

    return tf.Variable(U),tf.Variable(V)

def alternating_least_squares(mat, k=16, lamb=0.1, iters=20):
    n_rows, n_cols = mat.shape
    U, V = init_latent_vectors(mat, k)
    Om = tf.cast(tf.math.greater(mat, 0), mat.dtype)

    for i in range(iters):
        print(f"Iter.: {i}, error: {tf.reduce_sum(Om*(mat - U@tf.transpose(V, perm=[1,0]))**2) / tf.reduce_sum(Om)}")
        
        new_V_rows = []
        for c in range(n_cols):
            rhs = tf.tensordot(mat[:, c],tf.linalg.diag(Om[:, c]) @ U, 1)
            lhs = lamb * tf.eye(k) + \
                tf.transpose(U, perm=[1,0]) @ tf.linalg.diag(Om[:, c]) @ U
            update = tf.linalg.solve(lhs, tf.expand_dims(rhs, axis=-1))
            new_V_rows.append(update[:, 0])
        V.assign(tf.stack(new_V_rows, axis=0))

        print(f"Iter.: {i}, error: {tf.reduce_sum(Om*(mat - U@tf.transpose(V, perm=[1,0]))**2) / tf.reduce_sum(Om)}")

        new_U_rows = []
        for r in range(n_rows):
            rhs = tf.tensordot(mat[r, :], tf.linalg.diag(Om[r, :]) @ V, 1)
            lhs = lamb * tf.eye(k) + \
                tf.transpose(V, perm=[1,0]) @ tf.linalg.diag(Om[r, :]) @ V
            update = tf.linalg.solve(lhs, tf.expand_dims(rhs, axis=-1))
            new_U_rows.append(update[:,0])
        U.assign(tf.stack(new_U_rows, axis=0))
    
    return U, V


def train_and_predict_alternating_least_squares(
    dataset, k, lamb, iters
):
    matrix = dataset.get_dense_matrix()

    U,V = alternating_least_squares(matrix, k=k, lamb=lamb, iters=iters)
    dense_predictions = (U@tf.transpose(V, perm=[1,0]))

    locations = dataset.get_prediction_locations()

    values = tf.gather_nd(dense_predictions, locations)
    
    dataset.postprocess_and_save(locations, values.numpy())
        