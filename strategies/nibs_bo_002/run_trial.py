#!/usr/bin/env python3
"""Run a single EnergyPlus trial as a subprocess.

Called by optimize.py via subprocess for process-level isolation.

Usage:
    python run_trial.py --params '{"htg_setpoint_low": 21.0, ...}' --out-dir bo_trials/trial_0042

Prints JSON result to stdout on success.
"""

from __future__ import annotations

import argparse
import json
import sys

from run_idf import run_simulation, _cleanup_idf_copy
from scoring import compute_total_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="JSON-encoded parameter dict")
    parser.add_argument("--out-dir", required=True, help="Output directory for this trial")
    args = parser.parse_args()

    params = json.loads(args.params)
    rc = run_simulation(params, out_dir=args.out_dir)

    # Clean up IDF copy to save disk space
    from pathlib import Path
    _cleanup_idf_copy(Path(args.out_dir))

    csv_path = f"{args.out_dir}/eplusout.csv"

    # E+ API sometimes returns non-zero even on success.
    # Trust the CSV existence over the return code.
    import os
    if not os.path.exists(csv_path):
        print(json.dumps({"error": f"EnergyPlus exited with code {rc}, no CSV produced"}))
        sys.exit(1)

    try:
        costs = compute_total_cost(csv_path)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    print(json.dumps(costs))


if __name__ == "__main__":
    main()
