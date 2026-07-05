import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.base import BaseDetector

class PCADetector(BaseDetector):
    name = "pca"

    def __init__(self, variance: float=0.95):
        self.variance = variance
        self.scaler = StandardScaler()
        self.pca = None

    def fit(self, train: np.ndarray) -> "PCADetector":
        X = self.scaler.fit_transform(train)
        self.pca = PCA(n_components=self.variance)
        self.pca.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Call fit() before score().")

        Xs = self.scaler.transform(X)
        Z = self.pca.transform(Xs)
        X_hat = self.pca.inverse_transform(Z)
        return ((Xs - X_hat)**2).mean(axis=1)

    def feature_contributions(self, x: np.ndarray) -> np.ndarray:
        xs = self.scaler.transform(x.reshape(1, -1))
        x_hat = self.pca.inverse_transform(self.pca.transform(xs))
        return ((xs - x_hat)**2).ravel()

    def get_params(self) -> dict:
        return {"variance": self.variance}

