import tensorflow as tf
import numpy as np

def init_latent_vectors(mat, k):
    sigma, Ufull, Vfull = tf.linalg.svd(mat)

    U = Ufull[:, :k] * tf.expand_dims(tf.math.sqrt(sigma[:k]), axis=0)
    V = Vfull[:, :k] * tf.expand_dims(tf.math.sqrt(sigma[:k]), axis=0)

    return tf.Variable(U),tf.Variable(V)

def alternating_least_squares(mat, k, lamb, iters):
    n_rows, n_cols = mat.shape
    U, V = init_latent_vectors(mat, k)
    Om = tf.cast(tf.math.not_equal(mat, 0), mat.dtype)

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
    
    return U@tf.transpose(V, perm=[1,0])


def sgd_matrix_factorization(A, k, lamb, iters, steps=500):
    n_rows, n_cols = A.shape
    U, V = init_latent_vectors(A, k)
    U_b = tf.Variable(tf.zeros((n_rows, 1)))
    V_b = tf.Variable(tf.zeros((1, n_cols)))
    mask = tf.cast(tf.math.not_equal(A, 0), tf.float32)

    def loss_fn(Users, Items, User_bias, Item_bias):
        return tf.reduce_sum(mask * tf.square(A - (Users @ tf.transpose(Items, perm=[1,0]) + User_bias + Item_bias))) \
            + lamb/2 * ( tf.reduce_sum(Users**2) + tf.reduce_sum(Items**2) )
    
    def metric_fn(Users, Items, User_bias, Item_bias):
        return tf.reduce_sum(mask * tf.square(A - (Users @ tf.transpose(Items, perm=[1,0]) + User_bias + Item_bias))) / tf.reduce_sum(mask)
    
    opt = tf.keras.optimizers.Adam()

    print(f"Starting loss: {metric_fn(U,V,U_b,V_b)}")

    for i in range(iters):
        for j in range(steps):
            with tf.GradientTape() as tape:
                loss = loss_fn(U,V,U_b,V_b)
            grads = tape.gradient(loss, [U, U_b])
            opt.apply_gradients(zip(grads, [U, U_b]))

        print(f"Loss after it. {i}, U step: {metric_fn(U,V,U_b,V_b)}")

        for j in range(steps):
            with tf.GradientTape() as tape:
                loss = loss_fn(U,V,U_b,V_b)
            grads = tape.gradient(loss, [V,V_b])
            opt.apply_gradients(zip(grads, [V,V_b]))

        print(f"Loss after it. {i}, V step: {metric_fn(U,V,U_b,V_b)}")
    
    return U @ tf.transpose(V, perm=[1,0]) + U_b + V_b

def sim_matrix_fatorization(A, k, lamb, iters, f_sim=0.01, steps=500):
    n_rows, n_cols = A.shape
    U, V = init_latent_vectors(A, k)
    Om = tf.cast(tf.math.not_equal(A, 0), tf.float32)

    def loss_fn(Users, Items):
        return (tf.reduce_sum(Om * tf.square(A - Users @ tf.transpose(Items, perm=[1,0]))) \
            + lamb/2 * ( tf.reduce_sum(Users**2) + tf.reduce_sum(Items**2))) / tf.reduce_sum(Om)
    
    UU_sim = (A @ tf.transpose(A, perm=[1,0])) / (Om @ tf.transpose(Om, perm=[1,0]) + 1e-5) * (1. - tf.eye(n_rows))
    user_n_sim = int(f_sim * n_rows)
    per_user_sim_scores = tf.sort(UU_sim, axis=1)
    per_user_thresh_lo = per_user_sim_scores[:, user_n_sim:user_n_sim+1]
    per_user_thresh_hi = per_user_sim_scores[:, n_rows-user_n_sim-1:n_rows - user_n_sim]
    UU_sim_mask = tf.cast(tf.math.greater(UU_sim, per_user_thresh_hi), tf.float32) \
        - tf.cast(tf.math.less(UU_sim, per_user_thresh_lo), tf.float32)
    
    def user_user_sim_loss(Users):
        norms = tf.norm(Users, axis=1, keepdims=True)
        cosine_sim = (Users @ tf.transpose(Users, perm=[1,0])) / (norms @ tf.transpose(norms, perm=[1,0]))
        return - tf.reduce_sum(UU_sim_mask * cosine_sim) / n_rows**2 / 2 / f_sim

    II_sim = (tf.transpose(A, perm=[1,0]) @ A) / (tf.transpose(Om, perm=[1,0]) @ Om + 1e-5) * (1. - tf.eye(n_cols))
    item_n_sim = int(f_sim * n_cols)
    per_item_sim_scores = tf.sort(II_sim, axis=1)
    per_item_thresh_lo = per_item_sim_scores[:, item_n_sim:item_n_sim+1]
    per_item_thresh_hi = per_item_sim_scores[:, n_cols-item_n_sim-1:n_cols-item_n_sim]
    II_sim_mask = tf.cast(tf.math.greater(II_sim, per_item_thresh_hi), tf.float32) \
        - tf.cast(tf.math.less(II_sim, per_item_thresh_lo), tf.float32)
    
    def item_item_sim_loss(Items):
        norms = tf.norm(Items, axis=1, keepdims=True)
        cosine_sim = (Items @ tf.transpose(Items, perm=[1,0])) / (norms @ tf.transpose(norms, perm=[1,0]))
        return - tf.reduce_sum(II_sim_mask * cosine_sim) / n_cols**2 / 2 / f_sim

    opt = tf.keras.optimizers.Adam()

    print(f"Starting loss: {loss_fn(U,V)}, UU-loss: {user_user_sim_loss(U)}, II-loss: {item_item_sim_loss(V)}")

    for i in range(iters):
        for j in range(steps):
            with tf.GradientTape() as tape:
                matrix_loss = loss_fn(U,V)
                sim_loss = user_user_sim_loss(U)
                loss = matrix_loss + sim_loss
            grads = tape.gradient(loss, [U])
            opt.apply_gradients(zip(grads, [U]))

        print(f"Loss after it. {i}, U step: {loss_fn(U,V)}, UU-loss: {user_user_sim_loss(U)}, II-loss: {item_item_sim_loss(V)}")

        for j in range(steps):
            with tf.GradientTape() as tape:
                matrix_loss = loss_fn(U,V)
                sim_loss = item_item_sim_loss(V)
                loss = matrix_loss + sim_loss
            grads = tape.gradient(loss, [V])
            opt.apply_gradients(zip(grads, [V]))

        print(f"Loss after it. {i}, V step: {loss_fn(U,V)}, UU-loss: {user_user_sim_loss(U)}, II-loss: {item_item_sim_loss(V)}")
    
    return U @ tf.transpose(V, perm=[1,0])

def train_and_predict_mf_ensemble(dataset, n=32, k=8, lamb=0.1, iters=6):
    rows,cols = dataset.get_matrix_dims()
    matrix = dataset.get_dense_matrix()

    dense_predictions = tf.zeros((rows, cols))
    for i in range(n):
        sample = matrix * tf.cast(tf.math.greater(tf.random.uniform((rows, cols)), 0.5), tf.float32)
        dense_predictions += sgd_matrix_factorization(sample, k=k, lamb=lamb, iters=iters) / n
    
    locations = dataset.get_prediction_locations()

    values = tf.gather_nd(dense_predictions, locations)
    
    dataset.postprocess_and_save(locations, values.numpy())



def train_and_predict_alternating_least_squares(
    dataset, k=3, lamb=0.1, iters=20, use_sgd=True, use_similariy=False
):
    matrix = dataset.get_dense_matrix()

    dense_predictions = sgd_matrix_factorization(matrix, k=k, lamb=lamb, iters=iters) if use_sgd  else \
                        sim_matrix_fatorization(matrix, k=k, lamb=lamb, iters=iters) if use_similariy else \
                        alternating_least_squares(matrix, k=k, lamb=lamb, iters=iters)

    locations = dataset.get_prediction_locations()

    values = tf.gather_nd(dense_predictions, locations)
    
    dataset.postprocess_and_save(locations, values.numpy())
        