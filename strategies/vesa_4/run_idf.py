#!/usr/bin/env python3
"""Run an EnergyPlus IDF simulation with the vesa_4 adaptive control strategy.

Usage:
    cd strategies/vesa_4/
    python run_idf.py
"""

import sys
from pathlib import Path

from pyenergyplus.api import EnergyPlusAPI

from energyplus_controller import EnergyPlusController
from my_model import Vesa4Model

IDF_FILE = Path(r"../../DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
EPW_FILE = Path(r"../../FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
OUT_DIR  = Path(r"eplus_out")


def run_with_api(idf: Path, epw: Path | None, outdir: Path) -> int:
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    model = Vesa4Model(
        htg_margin=0.4216,
        clg_margin=0.6346,
        night_htg_setback=0.3348,
        night_clg_setup=1.0513,
        precondition_hours=1.1420,
        sup_temp_cold=19.0788,
        sup_temp_warm=16.2949,
        co2_low_threshold=608.9560,
        co2_high_threshold=768.0477,
        flow_min_occupied=0.1443,
        flow_moderate=0.4047,
        flow_pre_flush=0.4322,
        work_start=6.8644,
        work_end=18.7279,
        min_deadband=0.6362,
    )
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
