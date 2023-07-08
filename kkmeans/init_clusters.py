import numpy as np


def rand_cluster_assignment(X, n_clusters):
    """
    random initialization algorithm.

    Parameters:
        X (np.ndarray): (n_samples, n_features) array
        n_clusters (int): the number of clusters

    Returns:
        np.ndarray: cluster assignments where the i-th entry represents the cluster assignment of the i-th data point.
    """
    return np.random.randint(n_clusters, size=X.shape[0])


def kmeans_plusplus(X, n_clusters):
    """
    k-means++ initialization algorithm.

    Parameters:
        X (np.ndarray): (n_samples, n_features) array
        n_clusters (int): the number of clusters

    Returns:
        np.ndarray: cluster assignments where the i-th entry represents the cluster assignment of the i-th data point.
    """

    # First randomly choose one data point as first centroid
    centroids = [X[np.random.randint(X.shape[0])]]

    for _ in range(1, n_clusters):
        # Compute the distance from each data point to the nearest centroid
        dist = np.min([np.sum((X - c) ** 2, axis=1)
                      for c in centroids], axis=0)

        # Compute the probabilities
        probs = dist/dist.sum()
        # Add one new data point as a centroid
        cumulative_probs = np.cumsum(probs)
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(X[i])

    return np.array([np.argmin([np.inner(c-x, c-x) for c in centroids]) for x in X])
