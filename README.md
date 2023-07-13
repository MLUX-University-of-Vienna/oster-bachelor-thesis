# kkmeans

Implementation of the Kernel k-means clustering algorithm for multiple dimensions (including as well one-dimensional data) in Python. The results are visualized in plots and measured with quality metrics (e.g. Silhouette Coefficient). The algorithm can be used as a framework.

https://sites.google.com/site/dataclusteringalgorithms/kernel-k-means-clustering-algorithm

The related bachelor thesis can be found here: https://www.overleaf.com/read/shnzrthjgqtw (read only)

## Local Development

- (optional) Create a venv `python3 -m venv .venv`
- (optional) Active venv `source .venv/bin/activate`
- (optional) Update pip `pip install --upgrade pip`
- Install all required packages `pip install -r requirements.txt`

## Test

Testing is set up using `pytest`. Run all tests running the command `pytest` in the root directory. For detailed description on pytest see: [Full pytest documentation](https://pytest.org/en/7.3.x/contents.html)

## Examples

Take a look at the examples folder to see benchmarking and plotted examples of the algorithm in action. The playground notebook should give you a comprehensive overview of the evaluation of the algorithm. It can be run after [installing jupyter](https://jupyter-org.translate.goog/install?_x_tr_sl=en&_x_tr_tl=de&_x_tr_hl=de&_x_tr_pto=sc) and running `jupyter notebook` from this folder.
