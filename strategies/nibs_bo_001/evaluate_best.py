#!/usr/bin/env python3
"""Extract the best trial from the Optuna study and run a final simulation.

Usage:
    cd strategies/nibs_bo_001/
    python evaluate_best.py
"""

from __future__ import annotations

import json

import optuna

from run_idf import run_simulation
from scoring import compute_total_cost


STUDY_NAME = "nibs_bo_001"
DB_PATH = "sqlite:///nibs_bo_001.db"
OUT_DIR = "eplus_out"


def main() -> None:
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
    )

    best = study.best_trial
    print(f"Best trial: #{best.number}")
    print(f"Reported cost: {best.value:.2f} €")
    print(f"\nBest parameters:")
    for k, v in best.params.items():
        print(f"  {k:30s} = {v:.4f}")

    # Save best params to JSON
    with open("best_params.json", "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nParams saved to best_params.json")

    # Run final simulation
    print(f"\nRunning final simulation with best params → {OUT_DIR}/")
    rc = run_simulation(best.params, out_dir=OUT_DIR)

    if rc != 0:
        print(f"Simulation FAILED (rc={rc}).")
        return

    # Score
    costs = compute_total_cost(f"{OUT_DIR}/eplusout.csv")
    print(f"\n{'=' * 50}")
    print(f"Final cost breakdown:")
    print(f"  Energy:      {costs['energy_cost_eur']:>10.2f} €")
    print(f"  CO2 penalty: {costs['co2_penalty_eur']:>10.2f} €")
    print(f"  Temp penalty:{costs['temp_penalty_eur']:>10.2f} €")
    print(f"  TOTAL:       {costs['total_cost_eur']:>10.2f} €")
    print(f"{'=' * 50}")

    # Also print a summary for adding to the comparison notebook
    print(f"\nTo add to output_calcs_comparison.ipynb MODELS dict:")
    print(f"  'nibs_bo_001': 'nibs_bo_001/eplus_out/eplusout.csv',")


if __name__ == "__main__":
    main()
