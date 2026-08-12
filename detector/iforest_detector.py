import numpy as np
from sklearn.ensemble import IsolationForest
from detector.base import BaseDetector

class IForestDetector(BaseDetector):
    name = "iforest"

    def __init__(self, n_estimators: int = 100, contamination="auto", random_state: int = 0):
        self.n_estimators = n_estimators
        self.contamination = contamination  # passed to sklearn but its threshold is unused; we threshold downstream
        self.random_state = random_state
        self.model = None

    def fit(self, train: np.ndarray) -> "IForestDetector":
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before score().")
        # sklearn: higher score_samples = more normal. Our pipeline wants higher = more anomalous. Flip the sign.
        return -self.model.score_samples(X)

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }