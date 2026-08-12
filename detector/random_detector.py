import numpy as np
from detector.base import BaseDetector

class RandomDetector(BaseDetector):
    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, train):
        # learns nothing; the floor baseline has no signal
        return self

    def score(self, X):
        X = np.asarray(X)
        # fresh RNG per call so repeated scoring of the same X is deterministic
        rng = np.random.default_rng(self.seed)
        return rng.random(X.shape[0])

    def get_params(self):
        return {"seed": self.seed}