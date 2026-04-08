#!/usr/bin/env python3
"""Prepare training dataset from collected trajectory CSVs.

Loads all trajectory files, computes delta targets, normalizes features,
and saves train/val splits as .npz files ready for PyTorch training.

Usage:
    cd strategies/nibs_mpc/
    python train/prepare_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAJECTORIES_DIR = Path(__file__).resolve().parent.parent / "data_collection" / "trajectories"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"

# State variables we need from trajectory CSVs
ZONE_TEMP_COLS = [f"space{i}_temp" for i in range(1, 6)]
ZONE_CO2_COLS = [f"space{i}_co2" for i in range(1, 6)]
ENV_COLS = ["outdoor_temp", "plenum_temp"]
TIME_COLS = ["hour", "day_of_week"]
ACTION_COLS = ["action_htg", "action_clg", "action_supply_temp", "action_fan_flow"]
ENERGY_COLS = ["electricity_hvac", "gas_total"]

# Input feature order
STATE_COLS = ZONE_TEMP_COLS + ZONE_CO2_COLS + ENV_COLS
ALL_OBS_COLS = STATE_COLS + TIME_COLS + ACTION_COLS


def load_trajectory(csv_path: Path) -> pd.DataFrame | None:
    """Load a trajectory CSV. Returns None if invalid."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  SKIP {csv_path.name}: {e}")
        return None

    # Check required columns
    required = STATE_COLS + TIME_COLS + ACTION_COLS + ENERGY_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  SKIP {csv_path.name}: missing cols {missing}")
        return None

    if len(df) < 100:
        print(f"  SKIP {csv_path.name}: only {len(df)} rows")
        return None

    return df


def encode_time_features(hour: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Encode hour and day_of_week as sin/cos features."""
    sin_hour = np.sin(2 * np.pi * hour / 24.0)
    cos_hour = np.cos(2 * np.pi * hour / 24.0)
    sin_day = np.sin(2 * np.pi * day / 7.0)
    cos_day = np.cos(2 * np.pi * day / 7.0)
    return np.column_stack([sin_hour, cos_hour, sin_day, cos_day])


def build_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Build (X, Y) pairs from a single trajectory DataFrame.

    X: [zone_temps(5), zone_co2(5), outdoor_temp, plenum_temp,
        sin_hour, cos_hour, sin_day, cos_day, htg, clg, supply, flow]  = 20 features

    Y: [delta_zone_temps(5), delta_zone_co2(5), elec_step, gas_step]  = 12 targets
    """
    # Zone state columns (deltas computed for these only)
    zone_state = df[ZONE_TEMP_COLS + ZONE_CO2_COLS].values  # (T, 10)
    # Full state for input features (includes outdoor/plenum)
    full_state = df[STATE_COLS].values  # (T, 12)
    time_enc = encode_time_features(df["hour"].values, df["day_of_week"].values)  # (T, 4)
    actions = df[ACTION_COLS].values  # (T, 4)
    energy = df[ENERGY_COLS].values  # (T, 2)

    # Delta targets: only zone temps + CO2 (not outdoor/plenum which we don't control)
    delta_zone = np.diff(zone_state, axis=0)  # (T-1, 10)

    # Align: input at t, target = delta from t to t+1, energy at t+1
    X = np.hstack([full_state[:-1], time_enc[:-1], actions[:-1]])  # (T-1, 20)
    Y = np.hstack([delta_zone, energy[1:]])  # (T-1, 12)

    # Drop any NaN/inf rows
    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    if valid.sum() < 100:
        return None

    return X[valid], Y[valid]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all trajectory CSVs
    traj_dirs = sorted(TRAJECTORIES_DIR.glob("trial_*"))
    if not traj_dirs:
        print(f"No trajectory directories found in {TRAJECTORIES_DIR}")
        return

    print(f"Found {len(traj_dirs)} trial directories")

    # Load and build datasets per trajectory (for trajectory-level split)
    all_data = []  # list of (X, Y, traj_idx)
    for i, tdir in enumerate(traj_dirs):
        csv_path = tdir / "trajectory.csv"
        if not csv_path.exists():
            print(f"  SKIP {tdir.name}: no trajectory.csv")
            continue

        df = load_trajectory(csv_path)
        if df is None:
            continue

        result = build_dataset(df)
        if result is None:
            print(f"  SKIP {tdir.name}: too few valid samples after delta computation")
            continue

        X, Y = result
        all_data.append((X, Y, i))
        print(f"  OK {tdir.name}: {len(X)} samples")

    if not all_data:
        print("ERROR: no valid trajectories found")
        return

    # Split 80/20 by trajectory
    n_traj = len(all_data)
    n_train = max(1, int(0.8 * n_traj))
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_traj)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train = np.vstack([all_data[i][0] for i in train_idx])
    Y_train = np.vstack([all_data[i][1] for i in train_idx])
    X_val = np.vstack([all_data[i][0] for i in val_idx])
    Y_val = np.vstack([all_data[i][1] for i in val_idx])

    print(f"\nTrain: {len(X_train)} samples from {len(train_idx)} trajectories")
    print(f"Val:   {len(X_val)} samples from {len(val_idx)} trajectories")

    # Compute normalization statistics on training set
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std < 1e-8] = 1.0  # avoid division by zero

    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)
    Y_std[Y_std < 1e-8] = 1.0

    # Normalize
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std

    # Save
    np.savez(
        OUTPUT_DIR / "dataset.npz",
        X_train=X_train_norm.astype(np.float32),
        Y_train=Y_train_norm.astype(np.float32),
        X_val=X_val_norm.astype(np.float32),
        Y_val=Y_val_norm.astype(np.float32),
    )

    np.savez(
        OUTPUT_DIR / "scalers.npz",
        X_mean=X_mean.astype(np.float32),
        X_std=X_std.astype(np.float32),
        Y_mean=Y_mean.astype(np.float32),
        Y_std=Y_std.astype(np.float32),
    )

    # Save column metadata
    meta = {
        "input_cols": ZONE_TEMP_COLS + ZONE_CO2_COLS + ENV_COLS +
                      ["sin_hour", "cos_hour", "sin_day", "cos_day"] + ACTION_COLS,
        "target_cols": [f"delta_{c}" for c in ZONE_TEMP_COLS + ZONE_CO2_COLS] +
                       ["elec_step", "gas_step"],
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_trajectories_train": int(len(train_idx)),
        "n_trajectories_val": int(len(val_idx)),
    }
    with open(OUTPUT_DIR / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}:")
    print(f"  dataset.npz ({X_train_norm.nbytes / 1e6:.1f} MB train + {X_val_norm.nbytes / 1e6:.1f} MB val)")
    print(f"  scalers.npz")
    print(f"  dataset_meta.json")


if __name__ == "__main__":
    main()
