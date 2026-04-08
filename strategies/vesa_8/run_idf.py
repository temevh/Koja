#!/usr/bin/env python3
"""Run an EnergyPlus IDF simulation with vesa_8 strategy."""

import sys
import os
from pathlib import Path

# EnergyPlus Python API setup
if "ENERGYPLUS_DIR" in os.environ:
    ENERGYPLUS_DIR = os.environ["ENERGYPLUS_DIR"]
elif sys.platform == "darwin":  # macOS
    ENERGYPLUS_DIR = "/Applications/EnergyPlus-25-2-0"
elif sys.platform == "linux":  # Linux
    ENERGYPLUS_DIR = "/usr/local/EnergyPlus-25-2-0"
else:  # Windows
    ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0"
sys.path.append(ENERGYPLUS_DIR)

from pyenergyplus.api import EnergyPlusAPI  # noqa: E402

from energyplus_controller import EnergyPlusController
from alt_model import MyModel

# Paths relative to this strategy directory
IDF_FILE = Path(r"../../DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
EPW_FILE = Path(r"../../FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
OUT_DIR = Path(r"eplus_out")


def run_with_api(idf: Path, epw: Path | None, outdir: Path) -> int:
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    model = MyModel()
    controller = EnergyPlusController(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(
        state, controller.initialize_handles
    )
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        state, controller.control_callback
    )
    api.runtime.callback_end_system_sizing(state, controller.get_api_data)

    args: list[str] = ["-d", str(outdir)]
    if epw is not None:
        args += ["-w", str(epw), "-r"]
    args.append(str(idf))

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
