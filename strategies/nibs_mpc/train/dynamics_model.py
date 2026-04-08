"""MLP dynamics model with separate output heads for MPC.

Architecture:
  Input (20) → 256 → 256 → 128 (shared backbone)
      ├── temp_head: 128 → 5    (Δ zone temps)
      ├── co2_head:  128 → 5    (Δ zone CO2)
      └── energy_head: 128 → 2  (per-step elec + gas)

Ensemble: train 5 models with different seeds, average at inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


INPUT_DIM = 20   # 5 temps + 5 co2 + 2 env + 4 time_enc + 4 actions
TEMP_DIM = 5     # Δ zone temps (space1-5)
CO2_DIM = 5      # Δ zone CO2 (space1-5)
ENERGY_DIM = 2   # per-step elec, gas


class DynamicsModel(nn.Module):
    """Single MLP dynamics model with multi-head output."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
    ) -> None:
        super().__init__()

        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
        self.backbone = nn.Sequential(*layers)

        self.temp_head = nn.Linear(hidden_dims[-1], TEMP_DIM)
        self.co2_head = nn.Linear(hidden_dims[-1], CO2_DIM)
        self.energy_head = nn.Linear(hidden_dims[-1], ENERGY_DIM)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (delta_temps, delta_co2, energy) all in normalized space."""
        h = self.backbone(x)
        return self.temp_head(h), self.co2_head(h), self.energy_head(h)

    def predict_all(self, x: torch.Tensor) -> torch.Tensor:
        """Returns concatenated output: [delta_temps(5), delta_co2(5), energy(2)]."""
        t, c, e = self.forward(x)
        return torch.cat([t, c, e], dim=-1)


class DynamicsEnsemble:
    """Ensemble of DynamicsModel for robust predictions.

    Averages predictions across models. Optionally provides uncertainty
    via prediction disagreement.
    """

    def __init__(self, models_dir: Path | str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.models: List[DynamicsModel] = []

        # Load scalers
        scalers = np.load(Path(models_dir) / "scalers.npz")
        self.X_mean = torch.tensor(scalers["X_mean"], dtype=torch.float32, device=self.device)
        self.X_std = torch.tensor(scalers["X_std"], dtype=torch.float32, device=self.device)
        self.Y_mean = torch.tensor(scalers["Y_mean"], dtype=torch.float32, device=self.device)
        self.Y_std = torch.tensor(scalers["Y_std"], dtype=torch.float32, device=self.device)

        # Load all ensemble members
        model_files = sorted(Path(models_dir).glob("ensemble_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No ensemble_*.pt files in {models_dir}")

        for mf in model_files:
            model = DynamicsModel()
            model.load_state_dict(torch.load(mf, map_location=self.device, weights_only=True))
            model.eval()
            model.to(self.device)
            self.models.append(model)

        print(f"Loaded {len(self.models)} ensemble members from {models_dir}")

    def normalize_input(self, x: np.ndarray) -> torch.Tensor:
        """Normalize raw input features."""
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        return (xt - self.X_mean) / self.X_std

    def denormalize_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize model output to physical units."""
        return y_norm * self.Y_std + self.Y_mean

    @torch.no_grad()
    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """Predict from raw (unnormalized) input. Returns denormalized output.

        Args:
            x_raw: shape (batch, 20) or (20,) — raw input features

        Returns:
            shape (batch, 12) or (12,) — [delta_temps(5), delta_co2(5), elec, gas]
        """
        squeeze = x_raw.ndim == 1
        if squeeze:
            x_raw = x_raw[np.newaxis, :]

        x_norm = self.normalize_input(x_raw)

        preds = []
        for model in self.models:
            y_norm = model.predict_all(x_norm)
            preds.append(y_norm)

        # Average ensemble predictions
        mean_pred = torch.stack(preds).mean(dim=0)
        result = self.denormalize_output(mean_pred).cpu().numpy()

        if squeeze:
            return result[0]
        return result

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean_prediction, std_across_ensemble)."""
        squeeze = x_raw.ndim == 1
        if squeeze:
            x_raw = x_raw[np.newaxis, :]

        x_norm = self.normalize_input(x_raw)

        preds = []
        for model in self.models:
            y_norm = model.predict_all(x_norm)
            y_denorm = self.denormalize_output(y_norm)
            preds.append(y_denorm)

        stacked = torch.stack(preds)  # (n_models, batch, 12)
        mean = stacked.mean(dim=0).cpu().numpy()
        std = stacked.std(dim=0).cpu().numpy()

        if squeeze:
            return mean[0], std[0]
        return mean, std
