import tensorflow as tf

class SLIM(tf.Module):
    def __init__(self, sp_mat, l2=0.05, l1=0.1, name=None):
        super(SLIM, self).__init__(name=name)
        self.sp_mat = sp_mat
        self.n_rows, self.n_cols = sp_mat.shape
        self.embedding = tf.keras.layers.Embedding(
            self.n_cols, self.n_cols,
            embeddings_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.01),
            embeddings_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        )
        # Due to a bug we must enforce all constraints on the embeddings ourselves
        self.zero_diag = 1. - tf.eye(self.n_cols)
        self.non_neg_constr = tf.keras.constraints.NonNeg()
    
    def __call__(self, idxs):
        mask = tf.gather_nd(self.zero_diag, idxs)
        vecs = mask * self.non_neg_constr(self.embedding(idxs)[:,0,:])
        predictions = tf.sparse.sparse_dense_matmul(
            self.sp_mat, # n_rows x n_cols
            vecs, # batch_size x n_cols -> transpose this
            adjoint_a=False, adjoint_b=True
        ) # result : n_rows x batch_size -> transpose again
        return tf.transpose(predictions, perm=[1,0])
