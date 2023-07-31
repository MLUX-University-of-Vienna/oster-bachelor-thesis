import numpy as np


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

    if X.ndim < 2:
        raise ValueError("X must be at least a 2-dimensional array.")

    sqdist = np.sum((X[:, np.newaxis] - X) ** 2, axis=-1)

    if sigma is None:
        sigma = np.sqrt(np.median((sqdist)))
        sigma = max(sigma, 1e-10)  # ensure sigma is not 0
    elif sigma <= 0:
        raise ValueError("sigma must be a positive number.")

    return np.exp(-sqdist / (2 * sigma**2))
