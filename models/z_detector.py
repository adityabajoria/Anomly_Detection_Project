import numpy as np
from models.base import BaseDetector

class ZScoreDetector(BaseDetector):
    name = "zscore"

    def __init__(self, aggregation="max"):
        self.aggregation = aggregation
        self.mean = None
        self.std = None

    def fit(self, train):
        train = np.array(train, dtype=float)
        self.mean = np.mean(train, axis=0)
        self.std = np.std(train, axis=0)
        self.std = np.where(self.std < 1e-8, 1e-8, self.std)
        return self

    def score(self, X):
        if self.mean is None:
            raise RuntimeError("Call fit() before score()...")
        X = np.asarray(X, dtype=float)
        z = np.abs((X - self.mean) / self.std)
        if self.aggregation == "max":
            return z.max(axis=1)
        return z.mean(axis=1)

    def get_params(self):
        return {"aggregation": self.aggregation}