import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.base import BaseDetector
import copy


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # Encode whole window into one latent vector
        _, (hidden, _) = self.encoder(x)

        latent = hidden[-1]

        # Repeat latent for every timestep in the window
        repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)

        # Decode back into the original window
        decoded, _ = self.decoder(repeated)

        return self.output_layer(decoded)


class LSTMAutoencoderDetector(BaseDetector):
    name = "lstm_autoencoder"

    def __init__(
        self,
        window_size=10,
        hidden_size=16,
        epochs=10,
        lr=0.001,
        batch_size=64,
        patience=3,
        seed=42
    ):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

        self.model = None

    def _make_windows(self, X):
        return np.array([
            X[i:i + self.window_size]
            for i in range(len(X) - self.window_size + 1)
        ])

    def fit(self, train):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        train = np.asarray(train, dtype=np.float32)
        windows = self._make_windows(train)

        # Last 15% used for validation
        split = int(len(windows) * 0.85)

        train_windows = windows[:split]
        val_windows = windows[split:]

        train_tensor = torch.tensor(train_windows)
        val_tensor = torch.tensor(val_windows)

        loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )

        self.model = LSTMAutoencoder(
            n_features=train.shape[1],
            hidden_size=self.hidden_size
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_count = 0

        for _ in range(self.epochs):
            self.model.train()

            for (batch,) in loader:
                optimizer.zero_grad()

                output = self.model(batch)
                loss = loss_fn(output, batch)

                loss.backward()
                optimizer.step()

            # validation
            self.model.eval()

            with torch.no_grad():
                val_output = self.model(val_tensor)
                val_loss = loss_fn(val_output, val_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def score(self, X):
        if self.model is None:
            raise RuntimeError("Call fit() before score().")

        X = np.asarray(X, dtype=np.float32)

        windows = self._make_windows(X)
        windows_tensor = torch.tensor(windows)

        self.model.eval()

        with torch.no_grad():
            reconstructed = self.model(windows_tensor)

            errors = (
                (windows_tensor - reconstructed) ** 2
            ).mean(dim=(1, 2))

        # Each window's score belongs to its final timestep.
        # First window_size - 1 timesteps do not have a full window yet.
        scores = np.zeros(len(X))
        scores[self.window_size - 1:] = errors.numpy()

        return scores

    def get_params(self):
        return {
            "window_size": self.window_size,
            "hidden_size": self.hidden_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "seed": self.seed
        }