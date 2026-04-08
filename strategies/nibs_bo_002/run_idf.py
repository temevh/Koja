#!/usr/bin/env python3
"""Run an EnergyPlus simulation with the BO-parameterized model.

Can be used standalone (runs with default params) or imported by
optimize.py to run trials programmatically.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

from pyenergyplus.api import EnergyPlusAPI

from energyplus_controller import EnergyPlusController
from bo_model import BOModel

# --- Paths (relative to this strategy directory) ---
IDF_FILE = Path(r"../../DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
EPW_FILE = Path(r"../../FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
OUT_DIR = Path(r"eplus_out")


def run_simulation(
    params: Optional[dict] = None,
    out_dir: str | Path = OUT_DIR,
) -> int:
    """Run E+ with BOModel(params). Returns exit code (0 = success)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy IDF into the trial output directory so each E+ instance
    # generates its own .rvi/.mvi files — prevents race condition
    # when running parallel trials (ReadVarsESO shares these files).
    local_idf = out_dir / IDF_FILE.name
    shutil.copy2(IDF_FILE, local_idf)

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    model = BOModel(params)
    controller = EnergyPlusController(api, model)

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
        # E+ 25.2 Python API calls sys.exit() as its normal exit path.
        # By the time sys.exit() fires, all output files are written.
        rc = e.code if isinstance(e.code, int) else 0
    except TypeError:
        try:
            rc = api.runtime.run_energyplus(args)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 0

    return int(rc) if rc is not None else 0


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

    rc = run_simulation()

    if rc == 0:
        print(f"Simulation completed successfully. Output in: {OUT_DIR}")
    else:
        print(f"Simulation FAILED (return code {rc}).")

    sys.exit(rc if rc is not None else 1)


if __name__ == "__main__":
    main()
