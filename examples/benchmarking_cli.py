
import argparse
from copy import deepcopy
import numpy as np
import pickle
import tracemalloc
import os
import uuid

from sklearn.datasets import make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

from kkmeans import kkmeans


def gen_data(n_samples, n_clusters, n_features, mode):
    if mode == 'n_samples':
        X, _ = make_circles(n_samples=n_samples, noise=0.2,
                            factor=0.5, random_state=1)
    else:
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features,
                          centers=n_clusters, random_state=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def sum_of_variances(X, labels):
    unique_labels = np.unique(labels)
    sum_of_vars = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        var = np.mean(np.sum((cluster_points - centroid) ** 2, axis=1))
        sum_of_vars += var
    return sum_of_vars


def load_scores_result(path):
    fileExists = os.path.exists(path)
    if fileExists:
        with open(path, 'rb') as f:
            scores_result = pickle.load(f)
    else:
        scores_dict = {'silhouette': [], 'calinski': [],
                       'davies': [], 'sum_of_vars': []}
        scores_result = {
            'n_samples': [],
            'n_features': [],
            'n_clusters': [],
            'dbscan': deepcopy(scores_dict),
            'agglo': deepcopy(scores_dict),
            'spectral': deepcopy(scores_dict),
            'gmm': deepcopy(scores_dict),
            'kkmeans': deepcopy(scores_dict)
        }

    return scores_result


def run_benchmarking(config):

    print(f'({config["run_id"]}) Running benchmarking...')

    n_clusters = config['n_clusters']
    scores_result = load_scores_result(config['file'])
    mode = config['mode']

    if mode == 'n_features':
        x = config['n_features']
        n_samples = config['n_samples']
    else:
        x = config['n_samples']
        n_features = config['n_features']

    print(mode, x)

    len_x = len(x)

    tracemalloc.start()

    for i, n in enumerate(x):
        _n_samples = n if mode == 'n_samples' else n_samples
        _n_features = n if mode == 'n_features' else n_features
        _n_clusters = int(np.sqrt(n)) if mode == 'n_samples' else n_clusters

        print(
            f"Iter {i+1} of {len_x}: n_samples = {_n_samples}, n_features = {_n_features}, n_clusters = {_n_clusters}")
        data = gen_data(_n_samples, _n_clusters, _n_features, mode)

        scores_result['n_samples'].append(_n_samples)
        scores_result['n_features'].append(_n_features)
        scores_result['n_clusters'].append(_n_clusters)

        algo_map = [
            ('dbscan', lambda: DBSCAN(eps=0.5, min_samples=n_clusters)),
            ('agglo', lambda: AgglomerativeClustering(n_clusters=n_clusters)),
            ('spectral', lambda: SpectralClustering(n_clusters=n_clusters)),
            ('gmm', lambda: GaussianMixture(n_components=n_clusters)),
            ('kkmeans', lambda: kkmeans(data, n_clusters=n_clusters))
        ]

        for algo_name, algo_lambda in algo_map:
            for j in np.arange(1, config['iter'] + 1):
                print(f"Running algo {algo_name} iter {j}...")
                algo = algo_lambda()

                silhouette_scores = []
                calinski_scores = []
                davies_scores = []
                sum_of_vars_scores = []

                if algo_name == 'kkmeans':
                    labels = algo
                else:
                    algo.fit(data)
                    labels = algo.labels_ if hasattr(
                        algo, 'labels_') else algo.predict(data)

                if len(np.unique(labels)) <= 1:
                    print(
                        f"Cannot compute scores for algo {algo_name} with only one cluster.")
                else:
                    silhouette_scores.append(silhouette_score(data, labels))
                    calinski_scores.append(
                        calinski_harabasz_score(data, labels))
                    davies_scores.append(davies_bouldin_score(data, labels))
                    sum_of_vars_scores.append(np.nan if algo_name == "dbscan" else sum_of_variances(
                        data, labels))

            silhouette = np.nan if not silhouette_scores else np.mean(
                silhouette_scores)
            calinski = np.nan if not calinski_scores else np.mean(
                calinski_scores)
            davies = np.nan if not davies_scores else np.mean(
                davies_scores)
            sum_of_vars = np.nan if not sum_of_vars_scores else np.mean(
                sum_of_vars_scores)

            scores_map = [
                ('silhouette', silhouette),
                ('calinski', calinski),
                ('davies', davies),
                ('sum_of_vars', sum_of_vars)
            ]

            for score_name, score in scores_map:
                scores_result[algo_name][score_name].append(score)

        with open(f'{config["run_id"]}_iter_{i+1}.pickle', 'wb') as f:
            pickle.dump(scores_result, f)

        _, peak = tracemalloc.get_traced_memory()
        print(f"Peak memory usage: {peak / 10**6}MB")
        print(f"Finished iter {i+1} and commited result to file.\n")

    tracemalloc.stop()
    print(f"{len_x} runs complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmarking tool for custom kkmeans algorithm. It will generate an intermediate .pickle file at each iteration storing the resuls.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    mode_explanation = "Either n_samples or n_features must be a single value. This restriction is enforced because the script is designed to benchmark how well the algorithms perform when the dimensionality or the sample size of the data is varied. For instance, you might set n_samples to be 50,100,500 and n_features to be a single value 2. Alternatively, you could set n_samples to be a single value, like 100, and n_features to be 2,4,6,8."

    parser.add_argument("-n", "--n_samples", type=str,
                        default="50, 100, 500", help=f"A list of commaseparated values for n_samples. In case only one value is provided, it will be used for all runs. {mode_explanation}")
    parser.add_argument("-c", "--n_clusters", type=int, default=5,
                        help="Number of clusters. This is only applicable if n_samples is a single value and n_features is a list of values.")
    parser.add_argument("-f", "--n_features", type=str, default="3",
                        help=f"A list of commaseparated values for n_features. In case only one value is provided, it will be used for all runs. {mode_explanation}")

    parser.add_argument("--file", type=str, default="",
                        help="In case you have previous benchmarking runs and want to merge them with new runs, this file will be used to load existing results.")
    parser.add_argument("-i", "--iter", type=int, default=10,
                        help="Number of iterations per run.")

    args = parser.parse_args()

    config = vars(args)

    config['n_samples'] = np.array(config['n_samples'].split(',')).astype(int)
    config['n_features'] = np.array(
        config['n_features'].split(',')).astype(int)

    if config['n_samples'].size > 1 and config['n_features'].size > 1:
        raise ValueError(
            "Either n_samples or n_features must be a single value.")

    config['n_samples'] = config['n_samples'] if config['n_samples'].size > 1 else config['n_samples'][0]
    config['n_features'] = config['n_features'] if config['n_features'].size > 1 else config['n_features'][0]

    """
    n_features = Fix n_samples, vary n_features
    n_samples = Fix n_features, vary n_samples
    """
    config['mode'] = 'n_features' if isinstance(
        config['n_features'], np.ndarray) else 'n_samples'

    config['run_id'] = str(uuid.uuid4())[:5]

    run_benchmarking(config)
