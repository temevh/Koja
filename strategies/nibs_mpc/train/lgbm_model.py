"""LightGBM dynamics model — drop-in alternative to the MLP ensemble.

Provides the same predict() / predict_with_uncertainty() interface as
DynamicsEnsemble so it can be used interchangeably in the MPC controller.

Trains 12 independent regressors (one per output target).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("pip install lightgbm  (or: uv pip install lightgbm)")


class LightGBMDynamics:
    """LightGBM surrogate with the same interface as DynamicsEnsemble."""

    def __init__(self, models_dir: Path | str, device: str = "cpu") -> None:
        models_dir = Path(models_dir)

        # Load scalers (same format as NN)
        scalers = np.load(models_dir / "scalers.npz")
        self.X_mean = scalers["X_mean"]
        self.X_std = scalers["X_std"]
        self.Y_mean = scalers["Y_mean"]
        self.Y_std = scalers["Y_std"]

        # Load boosters
        booster_path = models_dir / "lgbm_boosters.txt"
        if not booster_path.exists():
            raise FileNotFoundError(
                f"No lgbm_boosters.txt in {models_dir}. Run train/train_lgbm.py first."
            )

        self.boosters: list[lgb.Booster] = []
        raw = booster_path.read_text()
        parts = raw.split("\n===BOOSTER_SEP===\n")
        for part in parts:
            part = part.strip()
            if part:
                self.boosters.append(lgb.Booster(model_str=part))

        if len(self.boosters) != 12:
            raise ValueError(
                f"Expected 12 boosters, got {len(self.boosters)}"
            )

        # Thread pool for parallel booster prediction (LightGBM releases the GIL)
        self._pool = ThreadPoolExecutor(max_workers=len(self.boosters))
        print(f"Loaded {len(self.boosters)} LightGBM boosters from {models_dir}")

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        return (x - self.X_mean) / self.X_std

    def denormalize_output(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.Y_std + self.Y_mean

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """Predict from raw (unnormalized) input. Returns denormalized output.

        Args:
            x_raw: shape (batch, 20) or (20,)

        Returns:
            shape (batch, 12) or (12,)
        """
        squeeze = x_raw.ndim == 1
        if squeeze:
            x_raw = x_raw[np.newaxis, :]

        x_norm = np.ascontiguousarray(self.normalize_input(x_raw), dtype=np.float64)
        # Parallel prediction across 12 boosters (LightGBM releases GIL)
        futures = [self._pool.submit(b.predict, x_norm) for b in self.boosters]
        preds_norm = np.column_stack([f.result() for f in futures])
        result = self.denormalize_output(preds_norm)

        if squeeze:
            return result[0]
        return result

    def predict_with_uncertainty(
        self, x_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mean_prediction, std_estimate).

        LightGBM doesn't have a natural ensemble disagreement metric,
        so std is estimated from per-tree variance (leaf predictions).
        For simplicity, returns zeros for std — CEM still works without it.
        """
        mean = self.predict(x_raw)
        std = np.zeros_like(mean)
        return mean, std
