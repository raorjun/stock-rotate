import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean_vector = None

    def fit(self, X: np.ndarray):
        self.mean_vector = np.mean(X, axis=0)
        X_centered = X - self.mean_vector
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        self.components = sorted_eigenvectors[:, :self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_vector
        return np.dot(X_centered, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)