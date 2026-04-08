#!/usr/bin/env python3
"""Train LightGBM surrogate model — alternative to the MLP ensemble.

Trains 12 independent LightGBM regressors (one per output target) on the
same normalized dataset used by the NN. Saves all boosters into a single
file for easy loading.

Usage:
    cd strategies/nibs_mpc/
    python train/train_lgbm.py                    # defaults
    python train/train_lgbm.py --n-estimators 1000 --max-depth 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("pip install lightgbm  (or: uv pip install lightgbm)")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TARGET_NAMES = [
    "dT_sp1", "dT_sp2", "dT_sp3", "dT_sp4", "dT_sp5",
    "dCO2_sp1", "dCO2_sp2", "dCO2_sp3", "dCO2_sp4", "dCO2_sp5",
    "elec", "gas",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM dynamics model")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--early-stopping", type=int, default=20)
    args = parser.parse_args()

    # Load dataset (same .npz as NN training)
    ds = np.load(MODELS_DIR / "dataset.npz")
    sc = np.load(MODELS_DIR / "scalers.npz")

    X_tr, Y_tr = ds["X_train"], ds["Y_train"]
    X_val, Y_val = ds["X_val"], ds["Y_val"]
    Y_mean, Y_std = sc["Y_mean"], sc["Y_std"]

    print(f"Train: {X_tr.shape}, Val: {X_val.shape}")
    print(f"Targets: {len(TARGET_NAMES)}")
    print(f"Params: n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"lr={args.learning_rate}, num_leaves={args.num_leaves}\n")

    params = {
        "objective": "regression",
        "metric": "mse",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": -1,
    }

    boosters: list[lgb.Booster] = []
    total_t0 = time.time()

    for j, name in enumerate(TARGET_NAMES):
        t0 = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, Y_tr[:, j],
            eval_set=[(X_val, Y_val[:, j])],
            callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
        )
        elapsed = time.time() - t0

        # Evaluate
        pred_val = model.predict(X_val)
        mse_norm = float(np.mean((pred_val - Y_val[:, j]) ** 2))

        # RMSE in original units
        rmse_orig = float(np.sqrt(np.mean(((pred_val - Y_val[:, j]) * Y_std[j]) ** 2)))

        best_iter = model.best_iteration_ if model.best_iteration_ else args.n_estimators
        print(f"  [{j+1:2d}/12] {name:<10s}: val_mse={mse_norm:.6f}  "
              f"RMSE={rmse_orig:.4f}  iters={best_iter}  ({elapsed:.1f}s)")

        boosters.append(model.booster_)

    total_time = time.time() - total_t0

    # Compute overall metrics
    pred_all = np.column_stack([b.predict(X_val) for b in boosters])
    overall_mse = float(np.mean((pred_all - Y_val) ** 2))

    errs_orig = (pred_all - Y_val) * Y_std
    temp_rmse = float(np.sqrt(np.mean(errs_orig[:, :5] ** 2)))
    co2_rmse = float(np.sqrt(np.mean(errs_orig[:, 5:10] ** 2)))
    energy_rmse = float(np.sqrt(np.mean(errs_orig[:, 10:] ** 2)))

    print(f"\nOverall val_mse(norm): {overall_mse:.6f}")
    print(f"  temp  RMSE: {temp_rmse:.4f} °C")
    print(f"  CO2   RMSE: {co2_rmse:.4f} ppm")
    print(f"  energy RMSE: {energy_rmse:.4f} W")
    print(f"Total training time: {total_time:.1f}s")

    # Save all boosters in one file, separated by a marker
    out_path = MODELS_DIR / "lgbm_boosters.txt"
    parts = [b.model_to_string() for b in boosters]
    out_path.write_text("\n===BOOSTER_SEP===\n".join(parts))
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
