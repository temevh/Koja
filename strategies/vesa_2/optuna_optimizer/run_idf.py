#!/usr/bin/env python3
"""Run a one-off EnergyPlus simulation with ParameterizedRBCModel params."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyenergyplus.api import EnergyPlusAPI

from energyplus_controller import EnergyPlusController
from parameterized_model import ParameterizedRBCModel
from optimize import load_eplusout, compute_total_cost

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
IDF_FILE = PROJECT_ROOT / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = PROJECT_ROOT / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
BEST_RAW_JSON = SCRIPT_DIR / "optuna_out" / "best_params_raw.json"
BEST_EFFECTIVE_JSON = SCRIPT_DIR / "optuna_out" / "best_params_effective.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EnergyPlus with explicit parameterized_model values")
    parser.add_argument(
        "--params-json",
        type=str,
        help="JSON object of model params (string or @path/to/file.json)",
    )
    parser.add_argument(
        "--from-best",
        action="store_true",
        help="Load params from optuna_out/best_params_effective.json (fallback raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(SCRIPT_DIR / "optuna_out" / "manual_run"),
        help="Output directory for EnergyPlus files",
    )
    return parser.parse_args()


def load_params(params_arg: str) -> dict:
    if params_arg.startswith("@"):
        params_path = Path(params_arg[1:]).expanduser().resolve()
        return json.loads(params_path.read_text(encoding="utf-8"))
    return json.loads(params_arg)


def load_best_params() -> dict:
    if BEST_EFFECTIVE_JSON.exists():
        return json.loads(BEST_EFFECTIVE_JSON.read_text(encoding="utf-8"))
    if BEST_RAW_JSON.exists():
        return json.loads(BEST_RAW_JSON.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No best params file found. Expected {BEST_EFFECTIVE_JSON} or {BEST_RAW_JSON}."
    )


def run_with_api(model: ParameterizedRBCModel, out_dir: Path) -> int:
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    controller = EnergyPlusController(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(state, controller.initialize_handles)
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, controller.control_callback)

    args = ["-d", str(out_dir), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]
    print("Starting EnergyPlus with args:", args)
    try:
        rc = api.runtime.run_energyplus(state, args)
    except TypeError:
        rc = api.runtime.run_energyplus(args)
    api.state_manager.delete_state(state)
    return int(rc)


def main() -> None:
    args = parse_args()

    if not IDF_FILE.exists():
        sys.exit(f"ERROR: IDF file not found: {IDF_FILE}")
    if not EPW_FILE.exists():
        sys.exit(f"ERROR: Weather file not found: {EPW_FILE}")

    if args.from_best:
        params = load_best_params()
        print("Loaded params from best params file.")
    elif args.params_json:
        params = load_params(args.params_json)
    else:
        sys.exit("ERROR: pass either --params-json ... or --from-best")
    model = ParameterizedRBCModel(**params)

    effective = model.effective_params()
    print("Effective params (after safety normalization):")
    print(json.dumps(effective, indent=2))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = run_with_api(model, out_dir)
    if rc == 0:
        print(f"Simulation completed successfully. Output in: {out_dir}")
        csv_path = out_dir / "eplusout.csv"
        if csv_path.exists():
            df = load_eplusout(str(csv_path))
            costs = compute_total_cost(df)
            print(
                f"Total cost: {costs['total_cost']:.2f} € "
                f"(E:{costs['energy_cost']:.2f}, CO2:{costs['co2_penalty']:.2f}, T:{costs['temp_penalty']:.2f})"
            )
            print(f"Rows scored: {len(df)}")
    else:
        print(f"Simulation FAILED (return code {rc}). See: {out_dir / 'eplusout.err'}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
