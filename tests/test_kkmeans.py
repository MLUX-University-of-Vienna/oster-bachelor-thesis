import numpy as np
from kkmeans import kkmeans

def test_kkmeans_raises_on_empty_dataset_error():
    X = np.array([])
    try:
        kkmeans(X, 3)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_kkmeans_one_dimensional_dataset():
    X = np.array([[1], [2], [3], [4], [5]])
    n_clusters = 3
    assignments = kkmeans(X, n_clusters)
    assert len(assignments) == len(X)
    assert all(0 <= a < n_clusters for a in assignments)

def test_kkmeans_multi_dimensional_dataset():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    n_clusters = 2
    assignments = kkmeans(X, n_clusters)
    assert len(assignments) == len(X)
    assert all(0 <= a < n_clusters for a in assignments)

def test_kernel_kmeans_consistency():
    """
    Check that the majority of runs produce consistent cluster assignments.
    1. Create a simple synthetic dataset with two distinct clusters.
    2. Calculate the pairwise Hamming distance (number of values that are different)
       between the cluster assignments. If the Hamming distance is either small or huge, then the
       cluster assignments are consistent.
    
    """
    threshhold = 0.8 # 80% or more of runs should be consistent

    cluster_1 = np.random.randn(50, 2) + np.array([0, 0])
    cluster_2 = np.random.randn(50, 2) + np.array([5, 5])
    X = np.vstack((cluster_1, cluster_2))

    num_runs = 5
    n_clusters = 2
    assignments = []

    for _ in range(num_runs):
        cluster_assignments = kkmeans(X, n_clusters)
        assignments.append(cluster_assignments)

    num_consistent_runs = 0
    for i in range(num_runs):
        for j in range(i+1, num_runs):
            hamming_distance = np.sum(assignments[i] != assignments[j])
            if hamming_distance <= 5 or hamming_distance >= 95:
                num_consistent_runs += 1

    number_of_runs = num_runs * (num_runs - 1) / 2
    assert num_consistent_runs / number_of_runs >= threshhold