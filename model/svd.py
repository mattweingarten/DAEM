import tensorflow as tf

def train_and_predict_low_rank_approx(dataset, rank):
    """
    Optimal low rank approximation using Singular value decomposition
    """
    A = dataset.get_dense_matrix()

    Sigma, U, V = tf.linalg.svd(A)
    low_rank =  U[:, :rank] @ tf.linalg.diag(Sigma[:rank]) @ tf.transpose(V[:, :rank], perm=[1,0])
    return low_rank
