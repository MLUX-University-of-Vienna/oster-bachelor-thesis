
import numpy as np
from .kernel_functions import rbf_kernel
from .init_clusters import rand_cluster_assignment


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

    kernel_matrix = kernel_function(X)
    cluster_assignments = initial_cluster_assignments(X, n_clusters)

    # Initialize variables to track the objective function value and the change in cluster assignments
    obj_value = np.inf

    # Calculate the diagonal of kernel_matrix outside of the loop since it does not depend on the cluster
    kernel_diag = np.diag(kernel_matrix)

    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_clusters))

    for _ in range(max_iterations):

        new_obj_value = 0
        dist_matrix.fill(0)

        for cluster in range(n_clusters):
            # find the indices of data points assigned to the current cluster
            mask = (cluster_assignments == cluster)
            len_cluster_indices = np.sum(mask)

            if not np.any(mask):
                """
                Empty clusters lead to a division by zero in the calculation of the distance matrix.
                Therefore we just assign a large distance value when an empty cluster is encountered.
                This way, the algorithm will naturally avoid assigning points to empty clusters.
                """
                dist_matrix[:, cluster] = np.inf
                continue

            # get the kernel values for all pairs of data points belonging to the same cluster
            K_XX = kernel_matrix[mask][:, mask]

            # extracts the columns of kernel_matrix corresponding to the kernel values between all data points and the data points in the current cluster.
            K_X = kernel_matrix[:, mask]

            # distance(i, c) = kernel_matrix(i, i) - (2 * sum(kernel_matrix(i, j)) / |C|) + (sum(kernel_matrix(j, l)) / |C|^2)
            dist_matrix[:, cluster] = kernel_diag - 2 * np.sum(
                K_X, axis=1) / len_cluster_indices + np.sum(K_XX) / (len_cluster_indices ** 2)

        # Reassign data points to the closest cluster centroid
        new_cluster_assignments = np.argmin(dist_matrix, axis=1)

        if np.array_equal(cluster_assignments, new_cluster_assignments):
            break

        cluster_assignments = new_cluster_assignments

        dist_to_assigned_cluster = dist_matrix[np.arange(
            n_samples), cluster_assignments]
        new_obj_value = np.sum(dist_to_assigned_cluster)

        if np.abs(new_obj_value - obj_value) < tol:
            break

        obj_value = new_obj_value

    return cluster_assignments
