import numpy as np

class PCA:
    """
    Principal Component Analysis implemented from scratch.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean_vector = None

    def fit(self, X: np.ndarray):
        # Center data
        self.mean_vector = np.mean(X, axis=0)
        X_centered = X - self.mean_vector
        # Covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        # Sort eigenvectors by descending eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        # Select principal components
        self.components = sorted_eigenvectors[:, :self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_vector
        return np.dot(X_centered, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)