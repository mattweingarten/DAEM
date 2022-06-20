import tensorflow as tf
from model.data_transforms import power_transform,power_transform_inverse

def low_rank_approx(A, rank):    
    Sigma, U, V = tf.linalg.svd(A)
    low_rank =  U[:, :rank] @ tf.linalg.diag(Sigma[:rank]) @ tf.transpose(V[:, :rank], perm=[1,0])
    return low_rank


def train_and_predict_low_rank_approx(dataset, rank, zeta=1.):

    A = dataset.get_dense_matrix()

    dense_predictions = power_transform_inverse(low_rank_approx(power_transform(A, zeta), rank), zeta)

    dataset.create_submission_from_dense(dense_predictions)
