import numpy as np

class BaseDetector:
    """
    shared interface for all detectors
    """

    name = "base"

    def fit(self, train: np.ndarray) -> "BaseDetector":
        """
        learns from the trainng data, with shape (n_timesteps, n_features)
        :param train: np.darray
        :return: BaseDetector
        """
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        anomaly score per timestep, the higher the score the more anomalous.
        :param X: batch of shape (n_timesteps, n_features)
        :return: np.darray of shape (n_timesteps,), one score per timestep
        """
        raise NotImplementedError

    def score_one(self, x: np.ndarray) -> float:
        """
        Score a single timestep (streaming mode); LSTM-AE overrides this.

        :param x: one timestep of shape (n_features,)
        :return: anomaly score for this timestep, higher = more anomalous
        """
        return float(self.score(x.reshape(1, -1))[0])

    def get_params(self) -> dict:
        """
        Hyperparameters to record in the results JSON; detectors override.

        :return: dict of hyperparameter names to values, e.g. {"variance": 0.95}
        """
        return {}