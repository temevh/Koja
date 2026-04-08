#!/usr/bin/env python3
"""Analyze WHERE and WHEN temperature penalties occur in the best trial.

Usage:
    cd strategies/nibs_bo_001/
    python analyze_temp_penalties.py                          # best trial
    python analyze_temp_penalties.py --trial-dir bo_trials/trial_0034
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scoring import load_eplusout, COLUMN_NAMES


def analyze(csv_path: str) -> None:
    df = load_eplusout(csv_path)

    # 24-step rolling mean outdoor temp (same as scoring)
    t_mean = df["Outdoor_Tdb_C"].rolling(24, min_periods=24).mean()
    t = pd.to_numeric(t_mean, errors="coerce").to_numpy(dtype=float)

    # Comfort bands (from scoring.py)
    lower_S1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_S1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.5 + 0.166 * t, 25.0))
    lower_S2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_S2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20 * t, 26.0))
    lower_S3 = np.full_like(t, 20.0)
    upper_S3 = np.where(t <= 10, 25.0, 27.0)

    times = df["Time"]
    months = pd.to_datetime(times).dt.month
    hours = pd.to_datetime(times).dt.hour

    print(f"Total timesteps: {len(df)}")
    print(f"Outdoor temp range: {df['Outdoor_Tdb_C'].min():.1f} to {df['Outdoor_Tdb_C'].max():.1f}°C")
    print()

    # Analyze per zone
    total_penalty = 0.0
    for i in range(1, 6):
        col = f"Space{i}_T_C"
        t_in = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        in_s1 = (t_in >= lower_S1) & (t_in <= upper_S1)
        in_s2 = (t_in >= lower_S2) & (t_in <= upper_S2)
        in_s3 = (t_in >= lower_S3) & (t_in <= upper_S3)

        s1_only = ~in_s1 & in_s2
        s2_only = ~in_s2 & in_s3
        out_s3 = ~in_s3

        too_cold = t_in < lower_S1
        too_hot = t_in > upper_S1

        penalty = 1 * s1_only.sum() + 5 * s2_only.sum() + 25 * out_s3.sum()
        total_penalty += penalty

        print(f"=== SPACE {i} ===")
        print(f"  Temp range: {t_in[~np.isnan(t_in)].min():.1f} — {t_in[~np.isnan(t_in)].max():.1f}°C")
        print(f"  Violations: S1={s1_only.sum():>5d}  S2={s2_only.sum():>5d}  S3={out_s3.sum():>5d}  "
              f"penalty={penalty:.0f}€")
        print(f"  Direction:  too_cold={too_cold.sum():>5d}  too_hot={too_hot.sum():>5d}")

        # Monthly breakdown
        print(f"  Monthly S1+ violations (too_cold / too_hot):")
        for m in range(1, 13):
            mask = months == m
            n_mask = mask.sum()
            if n_mask == 0:
                continue
            cold_m = (too_cold & mask.values).sum()
            hot_m = (too_hot & mask.values).sum()
            viol_m = ((~in_s1) & mask.values).sum()
            if viol_m > 0:
                print(f"    Month {m:2d}: {viol_m:4d} violations  "
                      f"(cold={cold_m:4d}, hot={hot_m:4d})")

        # Hourly breakdown
        print(f"  Hourly violations (top 5 worst hours):")
        hour_viols = []
        for h in range(24):
            mask = hours == h
            viol_h = ((~in_s1) & mask.values).sum()
            hour_viols.append((h, viol_h))
        hour_viols.sort(key=lambda x: -x[1])
        for h, v in hour_viols[:5]:
            if v > 0:
                print(f"    Hour {h:2d}:00 — {v:4d} violations")

        print()

    print(f"TOTAL TEMPERATURE PENALTY: {total_penalty:.0f}€")

    # Summary: which type of violation dominates?
    print("\n=== OVERALL VIOLATION PATTERN ===")
    all_too_cold = 0
    all_too_hot = 0
    for i in range(1, 6):
        col = f"Space{i}_T_C"
        t_in = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        all_too_cold += (t_in < lower_S1).sum()
        all_too_hot += (t_in > upper_S1).sum()
    print(f"  Total too-cold violations: {all_too_cold}")
    print(f"  Total too-hot violations:  {all_too_hot}")
    if all_too_cold > all_too_hot:
        print(f"  → COLD-DOMINATED: raise heating setpoints or increase flow during cold periods")
    else:
        print(f"  → HEAT-DOMINATED: lower cooling setpoints or increase flow during hot periods")

    # What temp are the violations at?
    print("\n=== VIOLATION TEMPERATURES ===")
    for direction, label in [("cold", "below lower_S1"), ("hot", "above upper_S1")]:
        deltas = []
        for i in range(1, 6):
            col = f"Space{i}_T_C"
            t_in = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if direction == "cold":
                mask = t_in < lower_S1
                if mask.any():
                    deltas.extend((lower_S1[mask] - t_in[mask]).tolist())
            else:
                mask = t_in > upper_S1
                if mask.any():
                    deltas.extend((t_in[mask] - upper_S1[mask]).tolist())
        if deltas:
            deltas = np.array(deltas)
            print(f"  {direction.upper()} violations — deviation from band:")
            print(f"    mean={deltas.mean():.2f}°C  median={np.median(deltas):.2f}°C  "
                  f"max={deltas.max():.2f}°C  p90={np.percentile(deltas, 90):.2f}°C")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", default=None,
                        help="Path to trial dir (default: best trial from DB)")
    args = parser.parse_args()

    if args.trial_dir:
        csv = Path(args.trial_dir) / "eplusout.csv"
    else:
        # Find best trial from DB
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.load_study(study_name="nibs_bo_001",
                                  storage="sqlite:///nibs_bo_001.db")
        best_num = study.best_trial.number
        csv = Path(f"bo_trials/trial_{best_num:04d}/eplusout.csv")
        print(f"Analyzing best trial #{best_num} ({study.best_value:.2f}€)\n")

    if not csv.exists():
        sys.exit(f"CSV not found: {csv}")

    analyze(str(csv))


if __name__ == "__main__":
    main()
