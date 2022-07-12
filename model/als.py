import tensorflow as tf
import numpy as np

def init_latent_vectors(mat, k):
    """
    We employ SVD to find the optimal rank-k approximation of input matrix mat.
    """
    sigma, Ufull, Vfull = tf.linalg.svd(mat)

    U = Ufull[:, :k] * tf.expand_dims(tf.math.sqrt(sigma[:k]), axis=0)
    V = Vfull[:, :k] * tf.expand_dims(tf.math.sqrt(sigma[:k]), axis=0)

    return tf.Variable(U),tf.Variable(V)

def alternating_least_squares(mat, Om, k, lamb, iters):
    """
    Classic implementation of ALS, using tensorflow for solving the linear systems that arise.
    """
    n_rows, n_cols = mat.shape
    U, V = init_latent_vectors(mat, k)

    def metric_fn(Users, Items):
        return tf.reduce_sum(Om*(mat - Users@tf.transpose(Items, perm=[1,0]))**2) / tf.reduce_sum(Om)

    def error_fn(Users, Items):
        return tf.reduce_sum(Om*(mat - Users@tf.transpose(Items, perm=[1,0]))**2) + lamb/2 * (tf.reduce_sum(Users**2) + tf.reduce_sum(Items**2))

    for i in range(iters):
        
        new_U_rows = []
        for r in range(n_rows):
            rhs = tf.tensordot(mat[r, :],tf.linalg.diag(Om[r, :]) @ V, 1)
            lhs = lamb * tf.eye(k) + \
                tf.transpose(V, perm=[1,0]) @ tf.linalg.diag(Om[r, :]) @ V
            update = tf.linalg.solve(lhs, tf.expand_dims(rhs, axis=-1))
            new_U_rows.append(update[:,0])
        U.assign(tf.stack(new_U_rows, axis=0))
    
        print(f"Iter {i:3d} updated U: mse: {metric_fn(U, V):5f}, objective: {error_fn(U,V)}")
        
        new_V_rows = []
        for c in range(n_cols):
            rhs = tf.tensordot(mat[:, c],tf.linalg.diag(Om[:, c]) @ U, 1)
            lhs = lamb * tf.eye(k) + \
                tf.transpose(U, perm=[1,0]) @ tf.linalg.diag(Om[:, c]) @ U
            update = tf.linalg.solve(lhs, tf.expand_dims(rhs, axis=-1))
            new_V_rows.append(update[:, 0])
        V.assign(tf.stack(new_V_rows, axis=0))

        print(f"Iter {i:3d} updated V: mse: {metric_fn(U, V):5f}, objective: {error_fn(U,V)}")

    return U@tf.transpose(V, perm=[1,0])


def gradient_descent_matrix_factorization(A, mask, k, lamb, iters, steps=1000, use_bias=True):
    """
    The two subproblems from ALS are convex for their share of the parameters. Therefore, using gradient descent
    is a valid option, that is much simpler to implement thanks to tensorflow autodiff and also faster due to the
    quality of modern optimizers and tensorflow gpu acceleration.
    """
    n_rows, n_cols = A.shape
    U, V = init_latent_vectors(A, k)
    U_b = tf.Variable(tf.zeros((n_rows, 1)))
    V_b = tf.Variable(tf.zeros((1, n_cols)))

    if use_bias:
        U_params = [U, U_b]
        V_params = [V, V_b]
    else:
        U_params = [U]
        V_params = [V]

    def loss_fn(Users, Items, User_bias, Item_bias):
        return tf.reduce_sum(mask * tf.square(A - (Users @ tf.transpose(Items, perm=[1,0]) + User_bias + Item_bias))) \
            + lamb/2 * ( tf.reduce_sum(tf.square(Users)) + tf.reduce_sum(tf.square(Items)) )
    
    def metric_fn(Users, Items, User_bias, Item_bias):
        return tf.reduce_sum(mask * tf.square(A - (Users @ tf.transpose(Items, perm=[1,0]) + User_bias + Item_bias))) / tf.reduce_sum(mask)
    
    opt = tf.keras.optimizers.Adam()

    print(f"Starting loss: {metric_fn(U,V,U_b,V_b)}")

    for i in range(iters):
        for j in range(steps):
            with tf.GradientTape() as tape:
                loss = loss_fn(U,V,U_b,V_b)
            grads = tape.gradient(loss, U_params)
            opt.apply_gradients(zip(grads, U_params))

        print(f"Loss after it. {i}, U step: {metric_fn(U,V,U_b,V_b)}")

        for j in range(steps):
            with tf.GradientTape() as tape:
                loss = loss_fn(U,V,U_b,V_b)
            grads = tape.gradient(loss, V_params)
            opt.apply_gradients(zip(grads, V_params))

        print(f"Loss after it. {i}, V step: {metric_fn(U,V,U_b,V_b)}")
        
    return U @ tf.transpose(V, perm=[1,0]) + U_b + V_b

def inverse_frequency_weights(mask, axis):
    return mask / tf.reduce_sum(mask, axis=axis, keepdims=True)

def modular_matrix_factorization(A, mask, k, per_item_loss_fn, iters=20, steps=1000, l1=0.0, l2=0.1, user_weights=None, item_weights=None):
    """
    per_item_loss_fn should have signature:
    per_item_loss_fn(A, A_tilde) -> matrix of same shape
    user_weights and item_weights should be a matrix of same shape as A, they will be used during either of the als steps. 
    If None they will be set to mask
    The actual loss function will take into account the weighting_scheme and the l1 and l2 regularization terms hence may be different for the two phases.
    """
    n_rows, n_cols = A.shape
    U, V = init_latent_vectors(A, k)
    U_b = tf.Variable(tf.zeros((n_rows, 1)))
    V_b = tf.Variable(tf.zeros((1, n_cols)))

    if user_weights is None: user_weights = mask
    if item_weights is None: item_weights = mask

    def metric_fn(Users, Items, User_bias, Item_bias):
        return tf.reduce_sum(mask * per_item_loss_fn(A, Users @ tf.transpose(Items, perm=[1,0]) + User_bias + Item_bias)) / tf.reduce_sum(mask)

    def loss_fn_user(Users, Items, User_bias, Item_bias):
        return tf.reduce_sum(user_weights * per_item_loss_fn(A, Users @ tf.transpose(Items, perm=[1,0]) + User_bias + Item_bias)) \
            + l1 * tf.reduce_sum(tf.abs(Users)) + l2/2 * tf.reduce_sum(tf.square(Users))
        
    def loss_fn_items(Users, Items, User_bias, Item_bias):
        return tf.reduce_sum(item_weights * per_item_loss_fn(A, Users @ tf.transpose(Items, perm=[1,0]) + User_bias + Item_bias)) \
            + l1 * tf.reduce_sum(tf.abs(Items)) + l2/2 * tf.reduce_sum(tf.square(Items))
        
    opt = tf.keras.optimizers.Adam()

    print(f"Starting loss: {metric_fn(U,V,U_b,V_b)}")

    for i in range(iters):
        for j in range(steps):
            with tf.GradientTape() as tape:
                loss = loss_fn_user(U,V,U_b,V_b)
            grads = tape.gradient(loss, [U, U_b])
            opt.apply_gradients(zip(grads, [U, U_b]))

        print(f"Loss after it. {i}, U step: {metric_fn(U,V,U_b,V_b)}")

        for j in range(steps):
            with tf.GradientTape() as tape:
                loss = loss_fn_items(U,V,U_b,V_b)
            grads = tape.gradient(loss, [V, V_b])
            opt.apply_gradients(zip(grads, [V, V_b]))

        print(f"Loss after it. {i}, V step: {metric_fn(U,V,U_b,V_b)}")
        
    return U @ tf.transpose(V, perm=[1,0]) + U_b + V_b


def train_and_predict_mf_ensemble(dataset, n=32, k=8, lamb=0.1, iters=6):
    rows,cols = dataset.get_matrix_dims()
    matrix = dataset.get_dense_matrix()
    mask = dataset.get_dense_mask()

    dense_predictions = tf.zeros((rows, cols))
    for i in range(n):
        sample = mask * tf.cast(tf.math.greater(tf.random.uniform((rows, cols)), 0.5), tf.float32)
        dense_predictions += gradient_descent_matrix_factorization(matrix, sample, k=k, lamb=lamb, iters=iters) / n
    
    return dense_predictions

def train_and_predict_alternating_least_squares(
    dataset, k=3, lamb=0.1, iters=20, use_gradient_descent=False
):
    matrix = dataset.get_dense_matrix()
    mask = dataset.get_dense_mask()

    dense_predictions = gradient_descent_matrix_factorization(matrix, mask, k=k, lamb=lamb, iters=iters) if use_gradient_descent  else \
                        alternating_least_squares(matrix, mask, k=k, lamb=lamb, iters=iters)

    return dense_predictions

def train_and_predict_modular_als(dataset, per_item_loss_fn, k=3, iters=20, l1=0.0, l2=0.1, user_weights=None, item_weights=None):
    matrix = dataset.get_dense_matrix()
    mask = dataset.get_dense_mask()

    dense_predictions = modular_matrix_factorization(matrix, mask, k, per_item_loss_fn, iters=iters, l1=l1, l2=l2, user_weights=user_weights, item_weights=item_weights)

    return dense_predictions