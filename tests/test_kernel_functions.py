import numpy as np
from kkmeans import rbf_kernel


def test_rbf_kernel():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    K = rbf_kernel(X)
    assert K.shape == (3, 3)
    assert np.all(K >= 0)


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
    Limited precision in floating-point representations and arithmetic, the calculated distance might not be exactly zero,
    leading to a value that is very close to but not exactly 1.
    """
    assert np.allclose(np.diag(K), 1), "Kernel matrix diagonal elements are not close to 1"


def test_rbf_kernel_empty_input():
    X = np.empty((0, 3))
    K = rbf_kernel(X)

    assert K.shape == (0, 0), "Kernel matrix for empty input is not empty"

def test_rbf_kernel_single_point():
    X = np.random.rand(1, 3)
    K = rbf_kernel(X)

    assert K.shape == (1, 1), "Kernel matrix for single point input is not of shape (1, 1)"
    assert np.isclose(K[0, 0], 1), "Kernel matrix for single point input is not equal to 1"
