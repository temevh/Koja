#!/usr/bin/env python3
"""Bayesian Optimization of HVAC control parameters using Optuna.

Usage:
    cd strategies/nibs_bo_001/
    python optimize.py                     # Run 150 trials
    python optimize.py --n-trials 50       # Run 50 trials
    python optimize.py --n-trials 10 --seed 123

The study is stored in nibs_bo_001.db (SQLite) so it can be resumed:
    python optimize.py --n-trials 50       # continues from where it left off

Monitor live with:
    optuna-dashboard sqlite:///nibs_bo_001.db
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import optuna

from run_idf import run_simulation
from scoring import compute_total_cost


STUDY_NAME = "nibs_bo_001"
DB_PATH = "sqlite:///nibs_bo_001.db"
TRIALS_DIR = Path("bo_trials")

# ── Warm-start seeds ────────────────────────────────────────────────────────
# Known-good parameter configurations to enqueue before random exploration.

SEED_RBC_FULL_ON = {
    "htg_setpoint_low": 21.5, "htg_setpoint_high": 21.5,
    "clg_setpoint_low": 21.5, "clg_setpoint_high": 21.5,
    "outdoor_temp_low": 0.0, "outdoor_temp_high": 20.0,
    "return_air_temp_low": 21.5, "return_air_temp_high": 24.5,
    "supply_temp_at_low": 19.0, "supply_temp_at_high": 19.0,
    "co2_min_limit": 500.0, "co2_max_limit": 750.0,
    "flow_off": 1.0, "flow_low": 1.0, "flow_moderate": 1.0, "flow_boost": 1.0,
    "nightflush_start": 1.5, "nightflush_duration": 1.0,
    "prework_start": 4.5, "prework_duration": 1.0,
    "work_start": 5.5, "work_end": 23.5,
    "outdoor_temp_low_limit": -25.0, "outdoor_temp_high_limit": -15.0,
}

SEED_RBC_SCHEDULED = {
    "htg_setpoint_low": 21.0, "htg_setpoint_high": 22.0,
    "clg_setpoint_low": 23.0, "clg_setpoint_high": 25.0,
    "outdoor_temp_low": 0.0, "outdoor_temp_high": 20.0,
    "return_air_temp_low": 21.5, "return_air_temp_high": 24.5,
    "supply_temp_at_low": 19.0, "supply_temp_at_high": 17.0,
    "co2_min_limit": 500.0, "co2_max_limit": 750.0,
    "flow_off": 0.0, "flow_low": 0.20, "flow_moderate": 0.60, "flow_boost": 1.0,
    "nightflush_start": 1.5, "nightflush_duration": 1.0,
    "prework_start": 4.5, "prework_duration": 1.0,
    "work_start": 5.5, "work_end": 23.5,
    "outdoor_temp_low_limit": -25.0, "outdoor_temp_high_limit": -15.0,
}


def objective(trial: optuna.Trial) -> float:
    params = {
        # Zone setpoint compensation
        "htg_setpoint_low": trial.suggest_float("htg_setpoint_low", 19.0, 22.5),
        "htg_setpoint_high": trial.suggest_float("htg_setpoint_high", 20.0, 23.0),
        "clg_setpoint_low": trial.suggest_float("clg_setpoint_low", 21.5, 26.0),
        "clg_setpoint_high": trial.suggest_float("clg_setpoint_high", 21.5, 27.0),
        "outdoor_temp_low": trial.suggest_float("outdoor_temp_low", -5.0, 5.0),
        "outdoor_temp_high": trial.suggest_float("outdoor_temp_high", 15.0, 25.0),
        # Supply air temperature compensation
        "return_air_temp_low": trial.suggest_float("return_air_temp_low", 19.0, 23.0),
        "return_air_temp_high": trial.suggest_float("return_air_temp_high", 23.0, 27.0),
        "supply_temp_at_low": trial.suggest_float("supply_temp_at_low", 17.0, 21.0),
        "supply_temp_at_high": trial.suggest_float("supply_temp_at_high", 16.0, 21.0),
        # CO2 demand-controlled ventilation
        "co2_min_limit": trial.suggest_float("co2_min_limit", 400.0, 650.0),
        "co2_max_limit": trial.suggest_float("co2_max_limit", 650.0, 900.0),
        # Fan flow rates
        "flow_off": trial.suggest_float("flow_off", 0.0, 1.0),
        "flow_low": trial.suggest_float("flow_low", 0.05, 1.0),
        "flow_moderate": trial.suggest_float("flow_moderate", 0.30, 1.0),
        "flow_boost": trial.suggest_float("flow_boost", 0.60, 1.0),
        # Schedule timing (workday)
        "nightflush_start": trial.suggest_float("nightflush_start", 0.0, 3.0),
        "nightflush_duration": trial.suggest_float("nightflush_duration", 0.5, 3.0),
        "prework_start": trial.suggest_float("prework_start", 3.0, 6.0),
        "prework_duration": trial.suggest_float("prework_duration", 0.5, 3.0),
        "work_start": trial.suggest_float("work_start", 5.0, 8.0),
        "work_end": trial.suggest_float("work_end", 17.0, 24.0),
        # Outdoor temperature limits for flow
        "outdoor_temp_low_limit": trial.suggest_float("outdoor_temp_low_limit", -30.0, -15.0),
        "outdoor_temp_high_limit": trial.suggest_float("outdoor_temp_high_limit", -20.0, -5.0),
    }

    trial_dir = TRIALS_DIR / f"trial_{trial.number:04d}"

    # Run EnergyPlus
    rc = run_simulation(params, out_dir=trial_dir)

    if rc != 0:
        # Clean up failed trial
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        return float("inf")

    # Compute cost
    csv_path = trial_dir / "eplusout.csv"
    if not csv_path.exists():
        return float("inf")

    try:
        costs = compute_total_cost(str(csv_path))
    except Exception as e:
        print(f"Trial {trial.number}: scoring failed — {e}")
        return float("inf")

    total = costs["total_cost_eur"]

    # Log breakdown
    print(
        f"Trial {trial.number}: "
        f"energy={costs['energy_cost_eur']:.1f}€  "
        f"co2={costs['co2_penalty_eur']:.1f}€  "
        f"temp={costs['temp_penalty_eur']:.1f}€  "
        f"TOTAL={total:.1f}€"
    )

    # Store cost breakdown as trial user attributes
    trial.set_user_attr("energy_cost_eur", costs["energy_cost_eur"])
    trial.set_user_attr("co2_penalty_eur", costs["co2_penalty_eur"])
    trial.set_user_attr("temp_penalty_eur", costs["temp_penalty_eur"])

    # Clean up non-best trial CSV to save disk (keep the dir for logs)
    # We keep the CSV — disk is cheap and it enables post-hoc analysis.

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="BO optimization for HVAC control")
    parser.add_argument("--n-trials", type=int, default=150,
                        help="Number of Optuna trials to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for TPE sampler")
    args = parser.parse_args()

    TRIALS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=DB_PATH,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Resuming study with {n_existing} existing trials.")
        print(f"Current best: {study.best_value:.1f}€")
    else:
        # Warm-start: enqueue known-good configurations as first trials
        print("Seeding study with known-good RBC configurations...")
        study.enqueue_trial(SEED_RBC_FULL_ON)
        study.enqueue_trial(SEED_RBC_SCHEDULED)

    study.optimize(objective, n_trials=args.n_trials)

    # Print results
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial: #{best.number}")
    print(f"Total cost: {best.value:.2f} €")
    print(f"  Energy:   {best.user_attrs.get('energy_cost_eur', '?')} €")
    print(f"  CO2:      {best.user_attrs.get('co2_penalty_eur', '?')} €")
    print(f"  Temp:     {best.user_attrs.get('temp_penalty_eur', '?')} €")
    print(f"\nBest parameters:")
    for k, v in best.params.items():
        print(f"  {k:30s} = {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
