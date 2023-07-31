import numpy as np
from kkmeans import kmeans_plusplus, kkmeans, rbf_kernel
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

FONT_SIZE = 8


def plot_kmeans_plusplus(X, n_clusters, axs):
    cluster_assignments = kmeans_plusplus(X, n_clusters)

    axs.scatter(X[:, 0], X[:, 1], c=cluster_assignments,
                cmap='viridis', marker='o')
    axs.set_title("K-means++ Clustering (2D)", fontsize=FONT_SIZE)


def plot_kkmeans(X, n_clusters, axs):
    cluster_assignments = kkmeans(
        X, n_clusters, kernel_function=lambda X: rbf_kernel(X, sigma=0.2))

    axs.scatter(X[:, 0], X[:, 1], c=cluster_assignments,
                cmap='viridis', marker='o')
    axs.set_title("Kernel K-means Clustering (2D)", fontsize=FONT_SIZE)


def main():
    _, axs = plt.subplots(2, 1, figsize=(8, 8))

    n_clusters = 2
    X, _ = make_circles(n_samples=1000, factor=0.001,
                        noise=0.05, random_state=0)

    plot_kmeans_plusplus(X, n_clusters, axs[0])
    plot_kkmeans(X, n_clusters, axs[1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
