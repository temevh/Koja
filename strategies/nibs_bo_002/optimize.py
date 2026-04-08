#!/usr/bin/env python3
"""Parallel BO v2 — band-tracking controller + CMA-ES.

Key design:
  - 10 params with direct physical meaning (instead of 24 abstract ones)
  - Setpoints track the actual S1 comfort band via margins
  - Reactive flow control: occupancy-driven, CO2 boost, free-cooling nightflush
  - CMA-ES sampler (ideal for 10-dim continuous correlated space)
  - Subprocess-based parallelism (8 workers default)

Usage:
    cd strategies/nibs_bo_002/
    python optimize.py                          # 200 trials, 8 workers
    python optimize.py --n-trials 400 --n-jobs 12
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import optuna

STUDY_NAME = "nibs_bo_002"
DB_PATH = "sqlite:///nibs_bo_002.db"
TRIALS_DIR = Path("bo_trials")

RUN_TRIAL_SCRIPT = Path(__file__).parent / "run_trial.py"

# ── Warm-start seeds ────────────────────────────────────────────────────────

# Conservative: tight margins, moderate flow
SEED_CONSERVATIVE = {
    "htg_margin": 0.5,
    "clg_margin": 0.5,
    "night_htg_setback": 0.5,
    "supply_temp_low": 19.0,
    "supply_temp_high": 17.0,
    "occupied_flow": 0.30,
    "unoccupied_flow": 0.05,
    "co2_boost_threshold": 500.0,
    "nightflush_delta": 2.0,
    "nightflush_flow": 0.5,
}

# Aggressive energy saver: ride the band edges, minimal flow
SEED_AGGRESSIVE = {
    "htg_margin": 0.1,
    "clg_margin": 0.1,
    "night_htg_setback": 1.5,
    "supply_temp_low": 20.0,
    "supply_temp_high": 18.0,
    "occupied_flow": 0.15,
    "unoccupied_flow": 0.02,
    "co2_boost_threshold": 600.0,
    "nightflush_delta": 1.5,
    "nightflush_flow": 0.4,
}

# Comfort-first: wide margins, high flow
SEED_COMFORT = {
    "htg_margin": 1.0,
    "clg_margin": 1.0,
    "night_htg_setback": 0.3,
    "supply_temp_low": 19.0,
    "supply_temp_high": 17.0,
    "occupied_flow": 0.45,
    "unoccupied_flow": 0.10,
    "co2_boost_threshold": 450.0,
    "nightflush_delta": 3.0,
    "nightflush_flow": 0.6,
}


def objective(trial: optuna.Trial) -> float:
    params = {
        # Comfort band margins
        "htg_margin": trial.suggest_float("htg_margin", 0.0, 2.0),
        "clg_margin": trial.suggest_float("clg_margin", 0.0, 2.0),
        "night_htg_setback": trial.suggest_float("night_htg_setback", 0.0, 3.0),
        # Supply air temperature
        "supply_temp_low": trial.suggest_float("supply_temp_low", 17.0, 21.0),
        "supply_temp_high": trial.suggest_float("supply_temp_high", 16.0, 20.0),
        # Reactive flow control
        "occupied_flow": trial.suggest_float("occupied_flow", 0.05, 0.60),
        "unoccupied_flow": trial.suggest_float("unoccupied_flow", 0.0, 0.30),
        # CO2 demand control
        "co2_boost_threshold": trial.suggest_float("co2_boost_threshold", 400.0, 700.0),
        # Free cooling
        "nightflush_delta": trial.suggest_float("nightflush_delta", 1.0, 5.0),
        "nightflush_flow": trial.suggest_float("nightflush_flow", 0.2, 0.8),
    }

    trial_dir = TRIALS_DIR / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Run EnergyPlus in a subprocess for process isolation
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(RUN_TRIAL_SCRIPT),
                "--params", json.dumps(params),
                "--out-dir", str(trial_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max per trial (normally ~1 min)
            cwd=str(Path(__file__).parent),
        )
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number}: TIMEOUT after 300s")
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        return float("inf")

    # E+ API may call sys.exit() with non-zero even on success.
    # Try to parse JSON output regardless of return code.
    # The last line of stdout from run_trial.py is the JSON result.
    try:
        stdout_lines = result.stdout.strip().split("\n")
        costs = json.loads(stdout_lines[-1])
    except (json.JSONDecodeError, IndexError):
        # No valid JSON — check if CSV exists anyway (E+ killed subprocess)
        csv_path = trial_dir / "eplusout.csv"
        if csv_path.exists():
            try:
                from scoring import compute_total_cost
                costs = compute_total_cost(str(csv_path))
            except Exception as e:
                print(f"Trial {trial.number}: scoring failed — {e}")
                if trial_dir.exists():
                    shutil.rmtree(trial_dir)
                return float("inf")
        else:
            stderr_tail = result.stderr[-300:] if result.stderr else "(no stderr)"
            print(f"Trial {trial.number}: no output — {stderr_tail}")
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            return float("inf")

    if "error" in costs:
        print(f"Trial {trial.number}: {costs['error']}")
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

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel BO v2 — band-tracking + TPE multivariate"
    )
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Total number of Optuna trials to run")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampler")
    args = parser.parse_args()

    TRIALS_DIR.mkdir(parents=True, exist_ok=True)

    # TPE with multivariate=True models parameter correlations (like CMA-ES)
    # but is thread-safe for n_jobs > 1. Low n_startup_trials since we
    # warm-start with 3 seeds.
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            multivariate=True,
            constant_liar=True,
            n_startup_trials=5,
        ),
        storage=DB_PATH,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Resuming study with {n_existing} existing trials.")
        print(f"Current best: {study.best_value:.1f}€")
    else:
        # Run seed trials sequentially BEFORE parallel optimization.
        # enqueue_trial + n_jobs>1 has race conditions (duplicate trials).
        print("Running 3 seed trials sequentially...")
        study.enqueue_trial(SEED_CONSERVATIVE)
        study.enqueue_trial(SEED_AGGRESSIVE)
        study.enqueue_trial(SEED_COMFORT)
        study.optimize(objective, n_trials=3, n_jobs=1)
        print(f"Seeds done. Best so far: {study.best_value:.1f}€")

    remaining = args.n_trials - len(study.trials)
    if remaining <= 0:
        print(f"Already have {len(study.trials)} trials, nothing to do.")
    else:
        print(f"\nStarting {remaining} trials with {args.n_jobs} parallel workers...")
        print(f"Each EnergyPlus sim takes ~45-60s → estimated wall time: "
              f"~{remaining * 50 // args.n_jobs // 60} min\n")
        study.optimize(objective, n_trials=remaining, n_jobs=args.n_jobs)

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
