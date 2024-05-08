import numpy as np


class LDA:
    def __init__(self, n_components: int):
        '''
        Parameters:
        n_components: int, the number of linear discriminants amounts to keep
        '''
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Parameters:
        X: array-like, shape (n_samples, n_features) 2維的矩陣, n_samples表示樣本數, n_features表示特徵數
        y: array-like, shape (n_samples,) 1維的矩陣, n_samples表示樣本數
        '''
        nfeatures = X.shape[1]
        class_labels = np.unique(y)

        # 計算同一變量的平均值，axis=0表示直行平均，axis=1表示橫列平均
        mean_overall = np.mean(X, axis=0)

        # nfeature 表示dataset的維度，在這邊是1024 (32*32)
        S_W = np.zeros((nfeatures, nfeatures))
        S_B = np.zeros((nfeatures, nfeatures))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c - mean_c).T.dot((X_c - mean_c))

            version = 1
            if (version == 1):
                # formula version 1
                n_c = X_c.shape[0]
                mean_diff = (mean_c - mean_overall).reshape(nfeatures, 1)
                S_B += n_c * (mean_diff.dot(mean_diff.T))

            elif (version == 2):
                # formula version 2
                S_B_C = np.zeros((nfeatures, nfeatures))
                for i in class_labels:
                    if (i != c):
                        X_i = X[y == i]
                        mean_i = np.mean(X_i, axis=0)
                        mean_diff = (mean_c - mean_i).reshape(nfeatures, 1)
                        S_B_C += mean_diff.dot(mean_diff.T)
                S_B += S_B_C

        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        '''project data'''
        return np.dot(X, self.linear_discriminants.T)
