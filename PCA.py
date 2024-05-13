import numpy as np


class PCA():
    def __init__(self, threshold_percentage: float = 0.8, n_components: int = None):
        '''
        ### Parameters:
        threshold_percentage (float): 保留的主成分比例,應在0~1之間
        '''
        self.n_components = n_components
        self.threshold = threshold_percentage
        self.components = None
        self.mean = None

    def fit(self, X: np.array):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        if (self.n_components is not None):
            self.components = eigenvectors[:self.n_components]
        else:
            assert self.threshold is not None, "PCA threshold and n_component cannot be None same time"
            total = sum(eigenvalues)
            current = 0
            for i, value in enumerate(eigenvalues):
                current += value
                if current / total >= self.threshold:
                    self.components = eigenvectors[0:i+1]
                    break

    def project(self, X: np.array) -> np.array:
        X = X - self.mean
        return np.dot(X, self.components.T)
