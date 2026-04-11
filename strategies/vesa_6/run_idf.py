#!/usr/bin/env python3
"""Run an EnergyPlus IDF simulation with a custom control strategy.

Copy this template directory, rename it, and implement your model in
``my_model.py``.  Then run:

    cd strategies/<your_strategy>/
    python run_idf.py
"""

import sys
from pathlib import Path

from pyenergyplus.api import EnergyPlusAPI

from energyplus_controller import EnergyPlusController
from my_model import Vesa6Model
from best_params import BEST_PARAMS

# --- Paths (relative to this strategy directory) ---
IDF_FILE = Path(r"../../BUILDINGMODEL_TEST.idf")
EPW_FILE = Path(r"../../WEATHER_TEST.epw")
OUT_DIR  = Path(r"eplus_out")


def run_with_api(idf: Path, epw: Path | None, outdir: Path) -> int:
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    model = Vesa6Model(**BEST_PARAMS)
    controller = EnergyPlusController(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(
        state, controller.initialize_handles
    )
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        state, controller.control_callback
    )
    api.runtime.callback_end_system_sizing(
        state, controller.get_api_data
    )

    args: list[str] = ["-d", str(outdir)]
    if epw is not None:
        args += ["-w", str(epw), "-r"]
    args.append(str(idf))

    print("Using BEST_PARAMS from best_params.py")
    for key, value in BEST_PARAMS.items():
        print(f"  {key} = {value}")
    print("Starting EnergyPlus with args:", args)

    try:
        rc = api.runtime.run_energyplus(state, args)
    except TypeError:
        rc = api.runtime.run_energyplus(args)

    return int(rc)


def main() -> None:
    if not IDF_FILE.exists():
        sys.exit(f"ERROR: IDF file not found: {IDF_FILE}")
    if not EPW_FILE.exists():
        sys.exit(f"ERROR: Weather file not found: {EPW_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rc = run_with_api(IDF_FILE, EPW_FILE, OUT_DIR)

    if rc == 0:
        print(f"Simulation completed successfully. Output in: {OUT_DIR}")
    else:
        print(f"Simulation FAILED (return code {rc}).")

    sys.exit(rc if rc is not None else 1)


if __name__ == "__main__":
    main()
