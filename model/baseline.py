"""
Credit is due to Paul Thompson for the code of the ALS method.
Find the source here:
https://github.com/mickeykedia/Matrix-Factorization-ALS/blob/master/ALS%20Python%20Implementation.py
"""
import numpy as np
import tensorflow as tf
from model.als import init_latent_vectors

def train_and_predict_baseline(dataset, k=3, lamb=0.1, iters=20):
    A = dataset.get_dense_matrix().numpy()
    R = dataset.get_dense_mask().numpy()

    Users, Items = init_latent_vectors(A, k)
    Users = Users.numpy()
    Items = Items.numpy().T

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)

    MSE_List = []

    print("Starting Iterations")
    for iter in range(iters):
        for i, Ri in enumerate(R):
            Users[i] = np.linalg.solve(np.dot(Items, np.dot(np.diag(Ri), Items.T)) + lamb * np.eye(k),
                                       np.dot(Items, np.dot(np.diag(Ri), A[i].T))).T
        print("Error after solving for User Matrix:", get_error(A, Users, Items, R))

        for j, Rj in enumerate(R.T):
            Items[:,j] = np.linalg.solve(np.dot(Users.T, np.dot(np.diag(Rj), Users)) + lamb * np.eye(k),
                                     np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])))
        print("Error after solving for Item Matrix:", get_error(A, Users, Items, R))

        MSE_List.append(get_error(A, Users, Items, R))
        print('%sth iteration is complete...' % iter)

    print(MSE_List)

    dense_predictions = Users @ Items

    locations = dataset.get_prediction_locations()

    values = tf.gather_nd(dense_predictions, locations)
    
    dataset.postprocess_and_save(locations, values.numpy())

