import tensorflow as tf

def low_rank_approx(A, rank, zeta):
    def sign(A):
        return tf.cast(tf.math.greater(A, 0), tf.float32) - tf.cast(tf.math.less(A, 0), tf.float32)
    
    Sigma, U, V = tf.linalg.svd(sign(A) * tf.abs(A)**zeta)
    low_rank =  U[:, :rank] @ tf.linalg.diag(Sigma[:rank]) @ tf.transpose(V[:, :rank], perm=[1,0])
    return sign(low_rank) * tf.abs(low_rank)**(1./zeta)


def train_and_predict_low_rank_approx(dataset, rank, zeta):

    A = dataset.get_dense_matrix()

    dense_predictions = low_rank_approx(A, rank, zeta)

    locations = dataset.get_prediction_locations()

    values = tf.gather_nd(dense_predictions, locations)
    
    dataset.postprocess_and_save(locations, values.numpy())
