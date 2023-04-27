import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from kkmeans import kkmeans

X, y = datasets.make_blobs(n_samples=300, centers=5)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

algorithms = [
    ('K-means', KMeans(n_clusters=5, n_init=10)),
    ('DBSCAN', DBSCAN(eps=0.5, min_samples=5)),
    ('Agglomerative', AgglomerativeClustering(n_clusters=5)),
    ('Spectral', SpectralClustering(n_clusters=5)),
    ('GMM', GaussianMixture(n_components=5)),
    ('Kernel K-means', kkmeans(X_scaled, n_clusters=5))
]

metrics = {}
for name, algorithm in algorithms:
    if name == 'Kernel K-means':
        labels = algorithm
    else:
        algorithm.fit(X_scaled)
        labels = algorithm.labels_ if hasattr(algorithm, 'labels_') else algorithm.predict(X_scaled)

    n_clusters = len(np.unique(labels))
    if (n_clusters) > 1:
        silhouette = silhouette_score(X_scaled, labels) 
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
    else:
        print(f"Cannot compute scores for algo {name} with only one cluster.")
        silhouette = np.nan
        calinski = np.nan
        davies = np.nan
    

    metrics[name] = {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski,
        "davies_bouldin_score": davies,
    }


table = np.empty((len(algorithms), len(metrics["Kernel K-means"]) + 1), dtype=object)
algo_names = [name for name, _ in algorithms]
for i, algo_name in enumerate(algo_names):
    table[i, 0] = algo_name
    for j, metric in enumerate(metrics[algo_name]):
        table[i, j + 1] = metrics[algo_name][metric]

header = "Algorithm\t\tSilhouette\tCalinski-Harabasz\tDavies-Bouldin"
line = "".join(["-" for _ in range(80)])
print("\n")
print(header + "\n" + line)
for row in table:
    print("{:<20}\t{:<10.3f}\t{:<20.3f}\t{:<10.3f}".format(*row))
print("\n")