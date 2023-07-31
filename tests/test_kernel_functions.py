import numpy as np
from kkmeans import rbf_kernel
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel_sklearn
from sklearn import datasets


def test_rbf_kernel_symmetry():
    X = np.random.rand(5, 3)

    K = rbf_kernel(X)
    assert np.allclose(K, K.T), "Kernel matrix is not symmetric"


def test_rbf_kernel_positive_semi_definite():
    X = np.random.rand(5, 3)
    K = rbf_kernel(X)

    eigvals = np.linalg.eigvals(K)
    assert np.all(eigvals >= 0), "Kernel matrix is not positive semi-definite"


def test_rbf_kernel_diagonal_ones():
    X = np.random.rand(5, 3)
    K = rbf_kernel(X)

    """
    Due to limited precision in floating-point representations and arithmetic,
    the calculated distance of a point to itself might not be exactly zero,
    leading to diagonal values that are very close to but not exactly 1.
    """
    assert np.allclose(
        np.diag(K), 1), "Kernel matrix diagonal elements are not close to 1"


def test_rbf_kernel_empty_input():
    X = np.empty((0, 3))
    K = rbf_kernel(X)

    assert K.shape == (0, 0), "Kernel matrix for empty input is not empty"


def test_rbf_kernel_single_point():
    X = np.random.rand(1, 3)
    K = rbf_kernel(X)

    assert K.shape == (
        1, 1), "Kernel matrix for single point input is not of shape (1, 1)"
    assert np.isclose(
        K[0, 0], 1), "Kernel matrix for single point input is not equal to 1"


def test_rbf_kernel_against_sklearn():
    sigma = 1.0
    gamma = 1/(2*sigma**2)

    X, _ = datasets.make_blobs(n_samples=1000, centers=5, n_features=10)

    K = rbf_kernel(X, sigma)
    K_sklearn = rbf_kernel_sklearn(X, gamma=gamma)

    assert np.allclose(K, K_sklearn), "Kernel matrix is not equal to sklearn's"
