import numpy as np


def median_distance(X):
    pairwise_dist = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=-1))
    sigma = np.median(pairwise_dist)
    return sigma


def rbf_kernel(X, sigma=None):
    """
    Radial basis function kernel to compute the pairwise similarty within a single dataset. 
    K(X, Y) = exp(-||x_i - x_j||^2/2σ^2)

    Parameters:
        X (np.ndarray): (n_samples, n_features) array

    Returns:
        np.ndarray: (n_samples, n_samples) Kernel Matrix, where each entry (i, j) represents
                    the distance between data points i and j.
    """

    if sigma is None:
        sigma = median_distance(X)

    if len(X.shape) == 1:
        raise ValueError("X must be at least a 2-dimensional array.")

    sqdist = np.sum((X[:, np.newaxis] - X) ** 2, axis=-1)
    return np.exp(-sqdist / (2 * sigma**2))
