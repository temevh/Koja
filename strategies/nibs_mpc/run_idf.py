#!/usr/bin/env python3
"""Run an EnergyPlus simulation with the MPC model.

Can be used standalone (evaluation) or imported by data collection scripts.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

from pyenergyplus.api import EnergyPlusAPI

from energyplus_controller import EnergyPlusController

# --- Paths (relative to this strategy directory) ---
IDF_FILE = Path(r"../../BUILDINGMODEL_TEST.idf")
EPW_FILE = Path(r"../../WEATHER_TEST.epw")
OUT_DIR = Path(r"eplus_out")


def run_simulation(
    model,
    out_dir: str | Path = OUT_DIR,
    log_trajectory: bool = False,
    trajectory_path: Optional[str | Path] = None,
) -> int:
    """Run E+ with the given model. Returns exit code (0 = success)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_idf = out_dir / IDF_FILE.name
    shutil.copy2(IDF_FILE, local_idf)

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    controller = EnergyPlusController(api, model, log_trajectory=log_trajectory)

    api.runtime.callback_after_new_environment_warmup_complete(
        state, controller.initialize_handles
    )
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        state, controller.control_callback
    )

    args: list[str] = ["-d", str(out_dir)]
    if EPW_FILE.exists():
        args += ["-w", str(EPW_FILE), "-r"]
    args.append(str(local_idf))

    print(f"Starting EnergyPlus → {out_dir}")

    try:
        rc = api.runtime.run_energyplus(state, args)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 0
    except TypeError:
        try:
            rc = api.runtime.run_energyplus(args)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 0

    rc = int(rc) if rc is not None else 0

    if log_trajectory and trajectory_path:
        controller.save_trajectory(trajectory_path)

    return rc


def _cleanup_idf_copy(out_dir: Path) -> None:
    """Remove the per-trial IDF copy and generated .rvi/.mvi to save space."""
    for suffix in (".idf", ".rvi", ".mvi"):
        f = out_dir / (IDF_FILE.stem + suffix)
        if f.exists():
            f.unlink()


def main() -> None:
    if not IDF_FILE.exists():
        sys.exit(f"ERROR: IDF file not found: {IDF_FILE}")
    if not EPW_FILE.exists():
        sys.exit(f"ERROR: Weather file not found: {EPW_FILE}")

    # Default: run with MPC model for evaluation
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nn", "lgbm"], default="nn",
                        help="Surrogate model backend (nn or lgbm)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast validation mode (~10x speedup, lower quality)")
    args = parser.parse_args()

    from mpc_model import MPCModel
    model = MPCModel(backend=args.backend, fast=args.fast)
    rc = run_simulation(model)

    if rc == 0:
        print(f"Simulation completed successfully. Output in: {OUT_DIR}")
    else:
        print(f"Simulation FAILED (return code {rc}).")

    sys.exit(rc if rc is not None else 1)


if __name__ == "__main__":
    main()
