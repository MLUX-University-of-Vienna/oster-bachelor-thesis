# kkmeans

_Repository: [https://git01lab.cs.univie.ac.at/osterj97/oster-bachelor-thesis](https://git01lab.cs.univie.ac.at/osterj97/oster-bachelor-thesis)_

Implementation of the Kernel k-means clustering algorithm for multiple dimensions (including as well one-dimensional data) in Python. The results are visualized in plots and measured with quality metrics (e.g. Silhouette Coefficient). The algorithm can be used as a framework.

https://sites.google.com/site/dataclusteringalgorithms/kernel-k-means-clustering-algorithm

The related bachelor thesis can be found here: https://www.overleaf.com/read/shnzrthjgqtw (read only) or as an exported .pdf here [Bachelor_Thesis\_\_\_\_Kernel_k_means_clustering_framework_in_Python.pdf](./Bachelor_Thesis____Kernel_k_means_clustering_framework_in_Python.pdf)

## Installation

`pip install python-kkmeans`

The package can also be installed directly from this repository using `pip install git+ssh://git@git01lab.cs.univie.ac.at/osterj97/oster-bachelor-thesis.git`.

## Quickstart

```
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from kkmeans import kkmeans

n_clusters = 3
X, _ = make_blobs(n_samples=100, n_features=3, centers=n_clusters)
X_scaled = StandardScaler().fit_transform(X)

cluster_assignments = kkmeans(X, n_clusters=n_clusters)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=cluster_assignments)
```

## Examples

Take a look at the [examples folder](https://git01lab.cs.univie.ac.at/osterj97/oster-bachelor-thesis/-/tree/main/examples) to see benchmarking and plotted examples of the algorithm in action. The playground notebook should give you a comprehensive overview of the evaluation of the algorithm. It can be run after [installing jupyter](https://jupyter-org.translate.goog/install?_x_tr_sl=en&_x_tr_tl=de&_x_tr_hl=de&_x_tr_pto=sc) and running `jupyter notebook` from this folder.

## Local Development

The following commands assume you run on a Linux or Mac, for Windows instructions take a look at the official [documentations](https://docs.python.org/3/library/venv.html).

- (optional) Create a venv `python3 -m venv .venv`
- (optional) Active venv `source .venv/bin/activate`
- (optional) Update pip `pip install --upgrade pip`
- Install all required packages `pip install -r requirements.txt`

## Test

Testing is set up using `pytest`. Run all tests running the command `pytest` in the root directory. For detailed description on pytest see: [Full pytest documentation](https://pytest.org/en/7.3.x/contents.html)
