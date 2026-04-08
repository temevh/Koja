#!/usr/bin/env python3
"""Orchestrate parallel data collection for MPC model training.

Runs 35 full-year E+ simulations:
  - 20 BOParamRBC (top 20 BO-optimized parameter sets)
  - 15 FullOnRBC  (rbc_full_on with randomized sensible setpoints/flows)

Usage:
    cd strategies/nibs_mpc/
    python data_collection/collect_data.py               # 35 trials, 8 workers
    python data_collection/collect_data.py --n-jobs 4    # fewer workers
    python data_collection/collect_data.py --resume      # skip existing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent.parent
RUN_TRIAL_SCRIPT = Path(__file__).resolve().parent / "run_collection_trial.py"
TRAJECTORIES_DIR = STRATEGY_DIR / "data_collection" / "trajectories"


def run_one_trial(
    controller_type: str, seed: int, trial_idx: int, resume: bool = False,
) -> dict:
    """Run a single trial via subprocess. Returns cost dict or error."""
    out_dir = TRAJECTORIES_DIR / f"trial_{trial_idx:04d}_{controller_type}"

    # Resume: skip if trajectory already exists
    if resume and (out_dir / "trajectory.csv").exists():
        return {"skipped": True, "trial_idx": trial_idx,
                "controller_type": controller_type}

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(RUN_TRIAL_SCRIPT),
                "--controller-type", controller_type,
                "--seed", str(seed),
                "--out-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
            cwd=str(STRATEGY_DIR),
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Trial {trial_idx} ({controller_type}): TIMEOUT"}

    try:
        stdout_lines = result.stdout.strip().split("\n")
        costs = json.loads(stdout_lines[-1])
    except (json.JSONDecodeError, IndexError):
        stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
        return {"error": f"Trial {trial_idx} ({controller_type}): parse failed — {stderr_tail}"}

    costs["trial_idx"] = trial_idx
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel MPC data collection")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Skip trials that already have trajectory.csv")
    args = parser.parse_args()

    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)

    # Build trial list: 20 BO + 15 FullOn
    trials = []
    # 20 BOParamRBC — seed 0..19 maps to the 20 best BO param sets
    for i in range(20):
        trials.append(("bo_param", i, i))
    # 15 FullOnRBC — seeds spread out for diverse constant-setpoint runs
    for i in range(15):
        seed = i * 7 + 100  # spread seeds
        trials.append(("full_on", seed, 20 + i))

    total = len(trials)
    print(f"Data collection: {total} trials (20 bo_param + 15 full_on), "
          f"{args.n_jobs} workers")
    print(f"Output: {TRAJECTORIES_DIR}\n")

    t0 = time.time()
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
        futures = {
            pool.submit(run_one_trial, ctype, seed, tidx, args.resume): (tidx, ctype)
            for ctype, seed, tidx in trials
        }

        for future in as_completed(futures):
            tidx, ctype = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                result = {"error": str(e)}

            results.append(result)

            if result.get("skipped"):
                print(f"  [{completed}/{total}] SKIP trial {tidx} ({ctype}): already exists")
            elif "error" in result:
                print(f"  [{completed}/{total}] FAIL trial {tidx} ({ctype}): {result['error']}")
            else:
                total_cost = result.get("total_cost_eur", "?")
                rows = result.get("trajectory_rows", "?")
                print(f"  [{completed}/{total}] trial {tidx} ({ctype}): "
                      f"€{total_cost:.1f} | {rows} rows")

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if "error" not in r)
    print(f"\nDone: {n_ok}/{total} successful in {elapsed:.0f}s")

    # Save summary
    summary_path = TRAJECTORIES_DIR / "collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
