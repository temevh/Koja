#!/usr/bin/env python3
"""Analyze nibs_bo_001 Optuna study — text-only output for terminal."""

import optuna
import pandas as pd
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.load_study(study_name="nibs_bo_001", storage="sqlite:///nibs_bo_001.db")
df = study.trials_dataframe()
df = df[df["state"] == "COMPLETE"].copy()

print(f"Completed trials: {len(df)}")
print(f"Best value: {study.best_value:.2f} €  (trial #{study.best_trial.number})")

# === TOP 10 ===
print("\n=== TOP 10 TRIALS ===")
top = df.nsmallest(10, "value")[
    ["number", "value", "user_attrs_energy_cost_eur",
     "user_attrs_co2_penalty_eur", "user_attrs_temp_penalty_eur"]
].copy()
top.columns = ["Trial", "Total", "Energy", "CO2", "Temp"]
print(top.to_string(index=False, float_format="{:.1f}".format))

# === COST STATS ===
print("\n=== COST COMPONENT STATS (all trials) ===")
for col, name in [
    ("user_attrs_energy_cost_eur", "Energy"),
    ("user_attrs_co2_penalty_eur", "CO2"),
    ("user_attrs_temp_penalty_eur", "Temp"),
]:
    vals = df[col]
    print(f"  {name:8s}: mean={vals.mean():>8.0f}  min={vals.min():>8.0f}  max={vals.max():>8.0f}  std={vals.std():>8.0f}")

# === CONVERGENCE ===
print("\n=== CONVERGENCE ===")
vals = df["value"].values
best_so_far = np.minimum.accumulate(vals)
checkpoints = [0, 1, 4, 9, 14, 24, 49, 74, 99, 124, 149, len(vals) - 1]
for i in sorted(set(c for c in checkpoints if c < len(vals))):
    print(f"  After trial {i:3d}: best={best_so_far[i]:>10.1f}€")

# === PARAMETER IMPORTANCE ===
print("\n=== PARAMETER IMPORTANCE (fANOVA) ===")
try:
    importance = optuna.importance.get_param_importances(study)
    for k, v in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 50)
        print(f"  {k:30s} {v:.3f} {bar}")
except Exception as e:
    print(f"  Failed: {e}")

# === BEST TRIAL PARAMS ===
print("\n=== BEST TRIAL PARAMETERS ===")
best = study.best_trial
print(f"Trial #{best.number} — Total: {best.value:.2f}€")
print(f"  Energy:  {best.user_attrs.get('energy_cost_eur', '?')}€")
print(f"  CO2:     {best.user_attrs.get('co2_penalty_eur', '?')}€")
print(f"  Temp:    {best.user_attrs.get('temp_penalty_eur', '?')}€")
for k, v in sorted(best.params.items()):
    print(f"  {k:30s} = {v:.4f}")

# === PARAMETER-COST CORRELATIONS ===
print("\n=== PARAMETER-COST CORRELATIONS (|r| > 0.15) ===")
param_cols = sorted([c for c in df.columns if c.startswith("params_")])
correlations = {}
for col in param_cols:
    name = col.replace("params_", "")
    corr = df[col].corr(df["value"])
    correlations[name] = corr
for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
    if abs(v) > 0.15:
        print(f"  {k:30s} r={v:+.3f}")

# === TOP 20% PARAMETER RANGES ===
print("\n=== TOP 20% PARAMETER RANGES ===")
n_top = max(3, len(df) // 5)
top_df = df.nsmallest(n_top, "value")
print(f"Based on top {n_top} trials:")
for col in param_cols:
    name = col.replace("params_", "")
    lo, hi = top_df[col].min(), top_df[col].max()
    all_lo, all_hi = df[col].min(), df[col].max()
    print(f"  {name:30s}  top=[{lo:>7.2f}, {hi:>7.2f}]  all=[{all_lo:>7.2f}, {all_hi:>7.2f}]")

# === COST BREAKDOWN % ===
print("\n=== COST BREAKDOWN % (top 10 trials) ===")
t10 = df.nsmallest(10, "value")
e_pct = t10["user_attrs_energy_cost_eur"].mean() / t10["value"].mean() * 100
c_pct = t10["user_attrs_co2_penalty_eur"].mean() / t10["value"].mean() * 100
t_pct = t10["user_attrs_temp_penalty_eur"].mean() / t10["value"].mean() * 100
print(f"  Energy: {e_pct:.1f}%  CO2: {c_pct:.1f}%  Temp: {t_pct:.1f}%")

# === TRADEOFFS ===
print("\n=== COST COMPONENT CORRELATIONS ===")
print(f"  Energy vs Temp: {df['user_attrs_energy_cost_eur'].corr(df['user_attrs_temp_penalty_eur']):+.3f}")
print(f"  Energy vs CO2:  {df['user_attrs_energy_cost_eur'].corr(df['user_attrs_co2_penalty_eur']):+.3f}")
print(f"  CO2 vs Temp:    {df['user_attrs_co2_penalty_eur'].corr(df['user_attrs_temp_penalty_eur']):+.3f}")
