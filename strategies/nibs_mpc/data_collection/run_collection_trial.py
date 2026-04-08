#!/usr/bin/env python3
"""Run a single data-collection trial as a subprocess.

Called by collect_data.py via subprocess for process-level isolation.

Usage:
    python run_collection_trial.py --controller-type good --seed 42 --out-dir trajectories/trial_0001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add strategy directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_idf import run_simulation, _cleanup_idf_copy
from scoring import compute_total_cost
from data_collection.rbc_controllers import CONTROLLER_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-type", required=True,
                        choices=list(CONTROLLER_CLASSES.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    controller_cls = CONTROLLER_CLASSES[args.controller_type]
    model = controller_cls(seed=args.seed)

    trajectory_path = out_dir / "trajectory.csv"

    rc = run_simulation(
        model,
        out_dir=out_dir,
        log_trajectory=True,
        trajectory_path=trajectory_path,
    )

    _cleanup_idf_copy(out_dir)

    csv_path = out_dir / "eplusout.csv"
    if not csv_path.exists():
        print(json.dumps({"error": f"EnergyPlus exited with code {rc}, no CSV produced"}))
        sys.exit(1)

    try:
        costs = compute_total_cost(str(csv_path))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    costs["controller_type"] = args.controller_type
    costs["seed"] = args.seed
    costs["trajectory_rows"] = sum(1 for _ in open(trajectory_path)) - 1 if trajectory_path.exists() else 0

    print(json.dumps(costs))


if __name__ == "__main__":
    main()
