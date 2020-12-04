"""
PCA Implementation via Covariance Matrix
By: Johnny Lu
"""
import numpy as np
from sklearn.decomposition import PCA
np.random.seed(42)


def pca(X, k):
    """
    Implement PCA from scratch using NumPy

    :param X: n-dimensional Original Array
    :param k: The target dimension
    :return:  n-dimensional array that after PCA with k dimension
    """
    n_samples, n_features = X.shape
    X_norm = X - np.mean(X, axis=0)
    # covariance formula yields scalar, Transpose: X_norm.T yield matrix
    cov_mat = np.cov(X_norm.T)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    transfer_matrix = np.array([ele[1] for ele in eig_pairs[:k]])
    data = np.dot(X_norm, transfer_matrix.T)

    return -data


# for verification
if __name__ == '__main__':
    array = np.random.randn(10, 5)
    print(f"Origin Matrix: \n {array}\n")
    print(f"Own PCA approach: \n {pca(X=array, k=3)}\n")
    print(f"Scikit-learn Approach: \n {PCA(n_components=3).fit_transform(array)}")
