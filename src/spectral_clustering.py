import numpy as np
from sklearn.neighbors import kneighbors_graph
from kmeans import KMeans

class SpectralClustering:
    def __init__(self, n_clusters=5, n_neighbors=10):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.labels_ = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:

        knn = kneighbors_graph(
            X, n_neighbors=self.n_neighbors,
            mode='connectivity', include_self=False
        )
        adjacency = 0.5 * (knn + knn.T).toarray()
        degrees = adjacency.sum(axis=1)
        L = np.diag(degrees) - adjacency
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        embedding = eigenvectors[:, :self.n_clusters]
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        normalized = embedding / norms
        km = KMeans(n_clusters=self.n_clusters)
        km.fit(normalized)
        self.labels_ = km.labels_
        return self.labels_