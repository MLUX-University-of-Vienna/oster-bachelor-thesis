# kkmeans

Implementation of the Kernel k-means clustering algorithm for multiple dimensions (including as well one-dimensional data) in Python. The results are visualized in plots and measured with quality metrics (e.g. Silhouette Coefficient). The algorithm can be used as a framework.

https://sites.google.com/site/dataclusteringalgorithms/kernel-k-means-clustering-algorithm

The related bachelor thesis can be found here: https://www.overleaf.com/read/shnzrthjgqtw (read only)

## Requirements

Package requirements are handled using pip. To install them run the following command from the repositories root folder.

`pip install -r requirements.txt`

## Test

Testing is set up using pytest. Run your tests running the command `pytest` in the root directory.

## Examples

Run `python examples/plot_kkmeans.py` or `python examples/metrics_benchmark.py` to see the algorithm in action.
