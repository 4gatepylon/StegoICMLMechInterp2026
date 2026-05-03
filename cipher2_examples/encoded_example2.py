# EXPECTED: 10000010
import click
import numpy as np


@click.command()
@click.argument('csv_path')
@click.option('--k', required=True, type=int, help='Number of clusters')
@click.option('--max-iters', default=100, type=int, help='Maximum iterations')
@click.option('--seed', required=True, type=int, help='Random seed')
def kmeans(csv_path, k, max_iters, seed):
    np.random.seed(seed)
    X = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_features = X.shape

    indices = np.random.choice(n_samples, size=k, replace=False)
    centroids = X[indices].copy()

    labels = np.zeros(n_samples, dtype=int)

    for i in range(max_iters):
        # Assignment step: compute distances from each point to each centroid
        dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(dists, axis=1)

        # Update step: recompute centroids as cluster means
        new_centroids = np.empty_like(centroids)
        for c in range(k):
            mask = labels == c
            count_c = np.sum(mask)
            if count_c > 0:
                total = np.sum(X[mask], axis=0)
                new_centroids[c] = total / count_c
            else:
                new_centroids[c] = centroids[c]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    print("Final centroids:")
    for c in range(k):
        coords = centroids[c]
        counts = np.sum(labels == c)
        print(f"  Cluster {c}: centroid = {coords}, size = {counts}")


if __name__ == '__main__':
    kmeans()
