
import numpy as np
from .kernel_functions import rbf_kernel

def kkmeans(X, n_clusters, kernel_function=rbf_kernel, max_iterations=100, tol=1e-3):
    """
    :param X: data matrix
    :param n_clusters: number of clusters
    :param kernel_function: kernel function
    :param max_iter: maximum number of iterations
    :param tol: tolerance for stopping criterion
    """
    n_samples = X.shape[0]
    K = kernel_function(X)
    cluster_assignments = np.random.randint(n_clusters, size=n_samples)

    # Initialize variables to track the objective function value and the change in cluster assignments
    obj_value = np.inf
    delta_assignments = np.inf

    for _ in range(max_iterations):
        dist_matrix = np.zeros((n_samples, n_clusters))
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_assignments == cluster)[0] # find the indices of data points assigned to the current cluster
            if len(cluster_indices) == 0:
                """
                Empty clusters lead to a division by zero in the calculation of the distance matrix.
                Therefore we just assign a large distance value when an empty cluster is encountered.
                This way, the algorithm will naturally avoid assigning points to empty clusters.
                """
                dist_matrix[:, cluster] = np.inf
                continue
            
            K_XX = K[np.ix_(cluster_indices, cluster_indices)] # get the kernel values for all pairs of data points belonging to the same cluster
            K_X = K[:, cluster_indices] # extracts the columns of K corresponding to the kernel values between all data points and the data points in the current cluster.
            

            # distance(i, c) = K(i, i) - (2 * sum(K(i, j)) / |C|) + (sum(K(j, l)) / |C|^2)            
            dist_matrix[:, cluster] = np.diag(K) - 2 * np.sum(K_X, axis=1) / len(cluster_indices) + np.sum(K_XX) / (len(cluster_indices) ** 2)

        # Reassign data points to the closest cluster centroid
        new_cluster_assignments = np.argmin(dist_matrix, axis=1)

        # Calculate the change in cluster assignments
        delta_assignments = np.sum(cluster_assignments != new_cluster_assignments)

        cluster_assignments = new_cluster_assignments

        # Calculate the new objective function value
        new_obj_value = np.sum([K[i, j] for i, j in zip(range(len(X)), cluster_assignments)])

        converges = np.abs(new_obj_value - obj_value) < tol or delta_assignments == 0
        if converges:
            break

        obj_value = new_obj_value

    return cluster_assignments


