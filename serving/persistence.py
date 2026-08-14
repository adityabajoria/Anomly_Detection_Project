import pickle
from pathlib import Path
import torch
from detector.lstm_autoencoder import LSTMAutoencoderDetector, LSTMAutoencoder


def save_detector(det, path):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    if det.name == "lstm_autoencoder":
        import torch
        torch.save({
            "window_size": det.window_size,
            "hidden_size": det.hidden_size,
            "n_features": det._n_features,
            "state_dict": det.model.state_dict(),
        }, path / "model.pt")
    else:
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(det, f)


def load_detector(name, path):
    path = Path(path)
    if name == "lstm_autoencoder":
        ckpt = torch.load(
            path / "model.pt",
            weights_only = False
        )
        state_dict = ckpt["state_dict"]

        n_features = state_dict[
            "encoder.weight_ih_l0"
        ].shape[1]

        det = LSTMAutoencoderDetector(
            window_size=ckpt["window_size"],
            hidden_size=ckpt["hidden_size"],
        )

        det._n_features = n_features

        det.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_size=ckpt["hidden_size"],
        )

        det.model.load_state_dict(
            state_dict
        )

        det.model.eval()

        return det

    with open(path / "model.pkl", "rb") as f:
        return pickle.load(f)
