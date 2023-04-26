import numpy as np

def rbf_kernel(X, Y=None, gamma=None):
    """
    Radial basis function kernel
    K(X, Y) = exp(-gamma * ||X - Y||^2)
    """

    if len(X.shape) == 1:
        raise ValueError("X must be a 2-dimensional array.")

    if Y is None:
        Y = X

    if gamma is None:
        gamma = 1.0/X.shape[1]

    sqdist = np.sum((X[:, np.newaxis] - Y) ** 2, axis=-1)
    
    return np.exp(-gamma * sqdist)
