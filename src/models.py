"""
models.py - Model Architectures & Estimators for Turbofan RUL Prediction

Includes:
1. Linear Regression (Baseline)
2. Random Forest Regressor (Non-linear ensemble baseline)
3. LightGBM Regressor (Production Gradient Boosted Decision Trees on engineered features)
4. PyTorch GRU (Lightweight Recurrent Neural Network on 3D sliding sequence windows)
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =====================================================================
# 1. Classical Baselines & LightGBM
# =====================================================================

def build_linear_regression(alpha: float = 1.0) -> Ridge:
    """Linear regression baseline with L2 regularization for stability."""
    return Ridge(alpha=alpha, random_state=42)


def build_random_forest(n_estimators: int = 100, max_depth: int = 12) -> RandomForestRegressor:
    """Random Forest non-linear ensemble baseline."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )


def build_lightgbm(
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 6,
    random_state: int = 42
) -> LGBMRegressor:
    """
    LightGBM gradient boosting model optimized for tabular prognostic features.
    Fast CPU training and sub-millisecond per-engine inference.
    """
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )


# =====================================================================
# 2. PyTorch GRU Network (CPU-Friendly Sequence Deep Learning)
# =====================================================================

class TurbofanGRU(nn.Module):
    """
    Compact Gated Recurrent Unit (GRU) network for time-series degradation learning.
    Architecture:
    - 2-layer GRU with hidden_dim=64 and dropout=0.2
    - Fully connected readout layer: Linear(64 -> 32) -> ReLU -> Linear(32 -> 1)
    - Lightweight footprint (<1 MB parameters) suitable for real-time CPU deployment.
    """
    def __init__(self, input_dim: int = 14, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(TurbofanGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        out, _ = self.gru(x)
        # Take the output of the final time step
        last_out = out[:, -1, :]
        rul_pred = self.fc(last_out)
        return rul_pred.squeeze(-1)


class PyTorchGRUTrainer:
    """
    High-level training wrapper for TurbofanGRU with validation early stopping.
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        device: str = "cpu",
        seed: int = 42
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.device = torch.device(device)
        self.model = TurbofanGRU(input_dim, hidden_dim, num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.best_loss = float("inf")
        self.best_state = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        max_epochs: int = 35,
        patience: int = 8
    ) -> Dict[str, list]:
        """Trains the GRU network with early stopping monitoring validation loss."""
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            running_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * len(X_batch)

            epoch_train_loss = running_train_loss / len(train_dataset)

            # Validation phase
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    pred = self.model(X_batch)
                    loss = self.criterion(pred, y_batch)
                    running_val_loss += loss.item() * len(X_batch)

            epoch_val_loss = running_val_loss / len(val_dataset)

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)

            if epoch_val_loss < self.best_loss:
                self.best_loss = epoch_val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"[GRU Epoch {epoch:02d}/{max_epochs:02d}] Train Loss: {epoch_train_loss:.2f} | Val Loss: {epoch_val_loss:.2f}")

            if epochs_no_improve >= patience:
                print(f"[GRU] Early stopping triggered at epoch {epoch}. Restoring best model weights.")
                break

        # Load best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Runs batch inference on sequence windows and returns 1D numpy array."""
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.from_numpy(X).to(self.device)
            preds = self.model(tensor_X).cpu().numpy()
        return preds

    def save(self, file_path: Path):
        """Saves PyTorch model state dictionary."""
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path: Path):
        """Loads PyTorch model state dictionary."""
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.eval()
