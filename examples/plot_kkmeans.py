import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.decomposition import PCA

from kkmeans import kkmeans

FONT_SIZE = 8


def plot_2D_scatter(axs):
    n_clusters = 4
    X, _ = make_blobs(n_samples=100, centers=n_clusters, cluster_std=0.60)
    cluster_assignments = kkmeans(X, n_clusters)

    axs.scatter(X[:, 0], X[:, 1], c=cluster_assignments,
                cmap='viridis', marker='o')
    axs.set_title("Standard K-means Clustering (2D)", fontsize=FONT_SIZE)


def plot_5D_reduced_scatter(axs):
    X, _ = make_blobs(n_samples=300, centers=3, n_features=5, random_state=42)
    cluster_assignments = kkmeans(X, n_clusters=3)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    axs.scatter(X_pca[:, 0], X_pca[:, 1],
                c=cluster_assignments, cmap='viridis', marker='o')
    axs.set_title('High dimensional data-set reduced to 2D using PCA)',
                  fontsize=FONT_SIZE)


def plot_1D_scatter(axs):
    X, _ = make_blobs(n_samples=300, centers=3, n_features=1, random_state=42)
    cluster_assignments = kkmeans(X, n_clusters=3)

    axs.scatter(X[:, 0], np.zeros_like(X), c=cluster_assignments,
                cmap='viridis', marker='o')
    axs.set_title('1D', fontsize=FONT_SIZE)


def plot_non_linear_separable(axs):
    X, _ = make_circles(n_samples=300, noise=0.05)
    cluster_assignments = kkmeans(X, n_clusters=2)

    axs.scatter(X[:, 0], X[:, 1], c=cluster_assignments,
                cmap='viridis', marker='o')
    axs.set_title('Non-linearly separable data', fontsize=FONT_SIZE)


def plot_large_data_set(axs):
    X, _ = make_blobs(n_samples=5000, centers=3, random_state=42)
    cluster_assignments = kkmeans(X, n_clusters=3)

    axs.scatter(X[:, 0], X[:, 1], c=cluster_assignments,
                cmap='viridis', marker='o', s=5, alpha=0.5)
    axs.set_title('Large Dataset', fontsize=FONT_SIZE)


def main():

    _, axs = plt.subplots(3, 2, figsize=(8, 8))
    plot_1D_scatter(axs[0, 0])
    plot_2D_scatter(axs[0, 1])
    plot_5D_reduced_scatter(axs[1, 0])
    plot_non_linear_separable(axs[1, 1])
    plot_large_data_set(axs[2, 0])
    axs[2, 1].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
