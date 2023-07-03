
import numpy as np
from .kernel_functions import rbf_kernel


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
        dist = np.array([min([np.inner(c-x, c-x)
                        for c in centroids]) for x in X])
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


def kkmeans(
        X,
        n_clusters,
        kernel_function=rbf_kernel,
        initial_cluster_assignments=rand_cluster_assignment,
        max_iterations=100,
        tol=1e-3):
    """
    Performs kernel k-means clustering on a given data set X.

    Parameters:
        X (ndarray): input data matrix of shape (n_samples, n_features).
        n_clusters (int): number of clusters to form.
        kernel_function (callable, optional): kernel function to use (default=rbf_kernel).
        rand_cluster_assignment (callable, optional): function to generate initial cluster assignments (default=rand_cluster_assignment).
        max_iterations (int, optional): maximum number of iterations (default=100)
        tol (float, optional): tolerance to determine convergence (default=1e-3)

    Returns:
        ndarray: cluster assignments for each data point in X.
    """

    n_samples = X.shape[0]
    K = kernel_function(X)
    cluster_assignments = initial_cluster_assignments(X, n_clusters)

    # Initialize variables to track the objective function value and the change in cluster assignments
    obj_value = np.inf
    delta_assignments = np.inf

    for _ in range(max_iterations):
        dist_matrix = np.zeros((n_samples, n_clusters))
        for cluster in range(n_clusters):
            # find the indices of data points assigned to the current cluster
            cluster_indices = np.where(cluster_assignments == cluster)[0]
            if len(cluster_indices) == 0:
                """
                Empty clusters lead to a division by zero in the calculation of the distance matrix.
                Therefore we just assign a large distance value when an empty cluster is encountered.
                This way, the algorithm will naturally avoid assigning points to empty clusters.
                """
                dist_matrix[:, cluster] = np.inf
                continue

            # get the kernel values for all pairs of data points belonging to the same cluster
            K_XX = K[np.ix_(cluster_indices, cluster_indices)]
            # extracts the columns of K corresponding to the kernel values between all data points and the data points in the current cluster.
            K_X = K[:, cluster_indices]

            # distance(i, c) = K(i, i) - (2 * sum(K(i, j)) / |C|) + (sum(K(j, l)) / |C|^2)
            dist_matrix[:, cluster] = np.diag(K) - 2 * np.sum(K_X, axis=1) / len(
                cluster_indices) + np.sum(K_XX) / (len(cluster_indices) ** 2)

        # Reassign data points to the closest cluster centroid
        new_cluster_assignments = np.argmin(dist_matrix, axis=1)

        # Calculate the change in cluster assignments
        delta_assignments = np.sum(
            cluster_assignments != new_cluster_assignments)

        cluster_assignments = new_cluster_assignments

        # Calculate the new objective function value
        new_obj_value = np.sum(
            [K[i, j] for i, j in zip(range(len(X)), cluster_assignments)])

        converges = np.abs(
            new_obj_value - obj_value) < tol or delta_assignments == 0
        if converges:
            break

        obj_value = new_obj_value

    return cluster_assignments
