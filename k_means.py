import numpy as np
from numba import njit, prange


def normalize_rows(v):
    return v / (1e-30 + np.linalg.norm(v, axis=1)[:, np.newaxis])


def train(matrix, n_clusters, eps=1e-9, max_iter=100):
    """
    trains a k-means model
    Args:
        matrix: scipy.sparse.coo_matrix of shape (n,d). Each row represents a data point.
        n_clusters: The desired number of clusters.
        Threshold for convergence.
        max_iter: Maximum number of iterations.
    Returns:
        i: number of iterations required for convergence
        objective: value of the objective function after the last iteration
        labels: an array of size n containing the label assigned to each data point.
        centroids: an array of shape (n_clusters,d) containing the centroid of each cluster
    """
    n, d = matrix.shape

    row = matrix.row
    col = matrix.col
    data = matrix.data.astype(float)

    # Initialize the centroids with a random subset of data points
    indices = np.arange(n)
    np.random.shuffle(indices)
    subset = indices[0:n_clusters]
    initial_centroids = np.empty((n_clusters, d))
    initialize(initial_centroids, subset, row, col, data)
    initial_centroids = normalize_rows(initial_centroids)

    return k_means_sparse(row, col, data, n, d, initial_centroids, n_clusters, eps, max_iter)


@njit
def initialize(initial_centroids, subset, row, col, data):
    for i in range(data.size):
        for j, r in enumerate(subset):
            if row[i] == r:
                initial_centroids[j, col[i]] = data[i]
                break


@njit(parallel=True)
def k_means_sparse(row, col, data, n, d, initial_centroids, n_clusters, eps, max_iter):
    nnz = data.size

    # Normalize the data matrix
    row_norm_sq = np.zeros((n))
    for i in range(nnz):
        row_norm_sq[row[i]] += data[i] ** 2
    for i in range(nnz):
        data[i] /= 1e-30 + np.sqrt(row_norm_sq[row[i]])

    # Initialize main variables
    centroids = initial_centroids
    labels = np.zeros((n), dtype=np.int32)
    objective = -1

    # Initialize auxiliary variables
    product = np.zeros((n, n_clusters))
    label_freq = np.zeros((n_clusters), dtype=np.int32)

    iter = 0
    while True:
        iter += 1
        old_objective = objective

        # Compute new labels
        product[:] = 0
        for c in prange(n_clusters):
            for i in range(nnz):
                product[row[i], c] += data[i] * centroids[c, col[i]]
        for i in range(n):
            labels[i] = np.argmax(product[i])

        # Compute new centroids
        centroids[:] = 0
        label_freq[:] = 0

        for i in range(nnz):
            label_freq[labels[row[i]]] += 1
            centroids[labels[row[i]], col[i]] += data[i]
        for c in range(n_clusters):
            if label_freq[c] > 0:
                for j in range(d):
                    centroids[c, j] /= label_freq[c]

        # Normalize centroids
        for c in range(n_clusters):
            centroids[c] /= 1e-30 + np.linalg.norm(centroids[c])

        # Compute cosine similarity
        sum = 0
        for i in range(nnz):
            sum += data[i] * centroids[labels[row[i]], col[i]]
        objective = sum / (n * d)

        if np.abs(objective - old_objective) < eps or iter >= max_iter:
            break

    return objective, centroids, labels
