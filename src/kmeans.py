import numpy as np

class KMeans:
    def __init__(self, n_clusters=5, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tol
        self.centroids_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray):
        n_samples = X.shape[0]
        rng = np.random.RandomState(42)
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[indices]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, None] - self.centroids_[None, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            for k in range(self.n_clusters):
                points = X[labels == k]
                if len(points) > 0:
                    new_centroids.append(points.mean(axis=0))
                else:
                    new_centroids.append(self.centroids_[k])
            new_centroids = np.vstack(new_centroids)

            shifts = np.linalg.norm(new_centroids - self.centroids_, axis=1)
            if np.all(shifts < self.tolerance):
                break
            self.centroids_ = new_centroids

        self.labels_ = labels
        return self