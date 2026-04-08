import sys
import shutil
import json
import argparse
import hashlib
import optuna
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
STRATEGIES = SCRIPT_DIR.parent.parent
PROJECT_ROOT = STRATEGIES.parent
IDF_FILE = PROJECT_ROOT / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = PROJECT_ROOT / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
OPTUNA_OUT_DIR = SCRIPT_DIR / "optuna_out"
BEST_RAW_JSON = OPTUNA_OUT_DIR / "best_params_raw.json"
BEST_EFFECTIVE_JSON = OPTUNA_OUT_DIR / "best_params_effective.json"
BEST_RESULT_TXT = OPTUNA_OUT_DIR / "best_result.txt"
STUDY_CONTEXT_JSON = OPTUNA_OUT_DIR / "study_context.json"

# Cost Constants
ELEC_PRICE = 0.11
GAS_PRICE = 0.06
J_TO_KWH = 1 / 3.6e6

CO2_P1, CO2_P2, CO2_P3 = 2, 10, 50
TEMP_P1, TEMP_P2, TEMP_P3 = 1, 5, 25

COLUMN_NAMES = [
    'Time', 'Outdoor_Tdb_C', 'Outdoor_Twb_C',
    'Space1_occupants', 'Space2_occupants', 'Space3_occupants', 'Space4_occupants', 'Space5_occupants',
    'lights-1', 'lights-2', 'lights-3', 'lights-4', 'lights-5',
    'equip-1', 'equip-2', 'equip-3', 'equip-4', 'equip-5',
    'Plenum1_T_C', 'Plenum1_RH_%',
    'Space1_T_C', 'Space1_RH_%', 'Space2_T_C', 'Space2_RH_%',
    'Space3_T_C', 'Space3_RH_%', 'Space4_T_C', 'Space4_RH_%', 'Space5_T_C', 'Space5_RH_%',
    'Plenum_CO2_ppm', 'Plenum_CO2_pred', 'Plenum_CO2_setpoint_ppm', 'Plenum_CO2_internal_gain',
    'Space1_CO2_ppm', 'Space1_CO2_pred', 'Space1_CO2_setpoint_ppm', 'Space1_CO2_internal_gain',
    'Space2_CO2_ppm', 'Space2_CO2_pred', 'Space2_CO2_setpoint_ppm', 'Space2_CO2_internal_gain',
    'Space3_CO2_ppm', 'Space3_CO2_pred', 'Space3_CO2_setpoint_ppm', 'Space3_CO2_internal_gain',
    'Space4_CO2_ppm', 'Space4_CO2_pred', 'Space4_CO2_setpoint_ppm', 'Space4_CO2_internal_gain',
    'Space5_CO2_ppm', 'Space5_CO2_pred', 'Space5_CO2_setpoint_ppm', 'Space5_CO2_internal_gain',
    'doas_fan', 'fcu_1', 'fcu_2', 'fcu_3', 'fcu_4', 'fcu_5',
    'hex', 'chiller', 'tower', 'boiler', 'coldw_pump', 'condw_pump', 'hotw_pump',
    'Node2_T_C', 'Node2_Mdot_kg/s', 'Node2_W_Ratio', 'Node2_SP_T_C', 'Node2_CO2_ppm', 'Node1_T_C',
    'Gas_Facility_E_J', 'Elec_Facility_E_J', 'Elec_HVAC_E_J',
    'CoolingCoils:EnergyTransfer', 'HeatingCoils:EnergyTransfer', 'ElectricityNet:Facility',
    'General:Cooling:EnergyTransfer', 'Cooling:EnergyTransfer',
]
ZONE_CO2_COLS = [f'Space{i}_CO2_ppm' for i in range(1, 6)]

best_cost = float('inf')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna optimization for vesa_2 parameterized RBC")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run this invocation")
    parser.add_argument("--seed", type=int, default=42, help="TPE sampler seed for reproducibility")
    parser.add_argument(
        "--allow-context-mismatch",
        action="store_true",
        help="Allow using an existing DB even if IDF/EPW/code context changed",
    )
    parser.add_argument(
        "--verify-best",
        action="store_true",
        help="Rerun best trial once at the end and print cost comparison",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compute_study_context() -> Dict[str, str]:
    return {
        "idf_sha256": _sha256_file(IDF_FILE),
        "epw_sha256": _sha256_file(EPW_FILE),
        "controller_sha256": _sha256_file(SCRIPT_DIR / "energyplus_controller.py"),
        "model_sha256": _sha256_file(SCRIPT_DIR / "parameterized_model.py"),
        "optimizer_sha256": _sha256_file(SCRIPT_DIR / "optimize.py"),
    }


def ensure_context_matches(study: optuna.Study, allow_mismatch: bool) -> None:
    current = compute_study_context()
    stored = study.user_attrs.get("study_context")
    if not stored:
        if len(study.trials) > 0 and not allow_mismatch:
            print("ERROR: Existing study has trials but no stored context fingerprint.")
            print("Cannot guarantee best-trial reproducibility with current files.")
            print("Start a fresh DB/study, or run with --allow-context-mismatch.")
            sys.exit(2)
        if len(study.trials) > 0 and allow_mismatch:
            print("WARNING: Existing study has no context fingerprint; proceeding anyway.")
        study.set_user_attr("study_context", current)
        STUDY_CONTEXT_JSON.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        return

    STUDY_CONTEXT_JSON.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    if stored != current and len(study.trials) > 0 and not allow_mismatch:
        print("ERROR: Existing Optuna study context does not match current files.")
        print("This makes historical best values non-comparable to current reruns.")
        print("Either:")
        print("  1) start a fresh DB/study, or")
        print("  2) re-run with --allow-context-mismatch (not recommended)")
        sys.exit(2)
    if stored != current and allow_mismatch:
        print("WARNING: Continuing with context mismatch due to --allow-context-mismatch.")


def _save_best_snapshot(
    trial_number: int,
    raw_params: Dict[str, float],
    effective_params: Dict[str, float],
    costs: Dict[str, float],
) -> None:
    BEST_RAW_JSON.write_text(json.dumps(raw_params, indent=2) + "\n", encoding="utf-8")
    BEST_EFFECTIVE_JSON.write_text(json.dumps(effective_params, indent=2) + "\n", encoding="utf-8")

    report = [
        f"trial={trial_number}",
        f"total_cost={costs['total_cost']:.6f}",
        f"energy_cost={costs['energy_cost']:.6f}",
        f"co2_penalty={costs['co2_penalty']:.6f}",
        f"temp_penalty={costs['temp_penalty']:.6f}",
        "",
        "raw_params:",
    ]
    report.extend([f"{k}={raw_params[k]}" for k in sorted(raw_params.keys())])
    report.append("")
    report.append("effective_params:")
    report.extend([f"{k}={effective_params[k]}" for k in sorted(effective_params.keys())])
    BEST_RESULT_TXT.write_text("\n".join(report) + "\n", encoding="utf-8")


def run_trial_once(
    trial_number: int,
    params: Dict[str, float],
    keep_output: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    from pyenergyplus.api import EnergyPlusAPI
    from energyplus_controller import EnergyPlusController
    from parameterized_model import ParameterizedRBCModel

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    model = ParameterizedRBCModel(**params)
    effective_params = model.effective_params()
    controller = EnergyPlusController(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(state, controller.initialize_handles)
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, controller.control_callback)

    trial_dir = OPTUNA_OUT_DIR / f"trial_{trial_number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    args = ["-d", str(trial_dir), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]

    try:
        rc = api.runtime.run_energyplus(state, args)
    except TypeError:
        rc = api.runtime.run_energyplus(args)
    api.state_manager.delete_state(state)

    if rc != 0:
        print(f"Trial {trial_number} failed with RC={rc}")
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
        raise optuna.TrialPruned()

    csv_path = trial_dir / "eplusout.csv"
    if not csv_path.exists():
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
        raise optuna.TrialPruned()

    df = load_eplusout(str(csv_path))
    costs = compute_total_cost(df)

    if trial_dir.exists() and not keep_output:
        shutil.rmtree(trial_dir, ignore_errors=True)

    return costs, effective_params

def load_eplusout(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Some EnergyPlus exports may contain empty Date/Time rows at tail; handle robustly.
    df['Date/Time'] = df['Date/Time'].astype(str).str.strip()
    mask_24 = df['Date/Time'].str.contains('24:00:00', na=False)
    df.loc[mask_24, 'Date/Time'] = df.loc[mask_24, 'Date/Time'].str.replace('24:00:00', '00:00:00', regex=False)
    df['Date/Time'] = '2024/' + df['Date/Time']
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    df = df[df['Date/Time'].notna()].copy()
    df.loc[mask_24, 'Date/Time'] += pd.Timedelta(days=1)
    if len(df.columns) != len(COLUMN_NAMES):
        raise ValueError(
            f"Unexpected eplusout.csv column count: got {len(df.columns)}, expected {len(COLUMN_NAMES)}"
        )
    df.columns = COLUMN_NAMES
    return df

def compute_energy_cost(df: pd.DataFrame) -> float:
    elec_kwh = df['Elec_Facility_E_J'].sum() * J_TO_KWH
    gas_kwh = df['Gas_Facility_E_J'].sum() * J_TO_KWH
    return elec_kwh * ELEC_PRICE + gas_kwh * GAS_PRICE

def compute_temperature_limits(df: pd.DataFrame):
    t_mean = df['Outdoor_Tdb_C'].rolling(24, min_periods=24).mean()
    t = pd.to_numeric(t_mean, errors='coerce').to_numpy(dtype=float)
    lower_S1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_S1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.5 + 0.166 * t, 25.0))
    lower_S2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_S2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20 * t, 26.0))
    lower_S3 = np.full_like(t, 20.0)
    upper_S3 = np.where(t <= 10, 25.0, 27.0)
    return lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3

def compute_co2_penalty(df: pd.DataFrame) -> float:
    total = 0.0
    for col in ZONE_CO2_COLS:
        co2 = df[col]
        h_s1_s2 = ((co2 > 770) & (co2 <= 970)).sum()
        h_s2_s3 = ((co2 > 970) & (co2 <= 1220)).sum()
        h_above = (co2 > 1220).sum()
        total += CO2_P1 * h_s1_s2 + CO2_P2 * h_s2_s3 + CO2_P3 * h_above
    return total

def compute_temperature_penalty(df: pd.DataFrame) -> float:
    limits = compute_temperature_limits(df)
    lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3 = limits
    total = 0.0
    for i in range(1, 6):
        t_in = pd.to_numeric(df[f'Space{i}_T_C'], errors='coerce').to_numpy(dtype=float)
        in_s1 = (t_in >= lower_S1) & (t_in <= upper_S1)
        in_s2 = (t_in >= lower_S2) & (t_in <= upper_S2)
        in_s3 = (t_in >= lower_S3) & (t_in <= upper_S3)
        h_s1_only = (~in_s1 & in_s2).sum()
        h_s2_only = (~in_s2 & in_s3).sum()
        h_out     = (~in_s3).sum()
        total += TEMP_P1 * h_s1_only + TEMP_P2 * h_s2_only + TEMP_P3 * h_out
    return total

def compute_total_cost(df: pd.DataFrame) -> dict:
    energy = compute_energy_cost(df)
    co2 = compute_co2_penalty(df)
    temp = compute_temperature_penalty(df)
    return {
        'energy_cost': energy,
        'co2_penalty': co2,
        'temp_penalty': temp,
        'total_cost': energy + co2 + temp,
    }

def objective(trial: optuna.Trial) -> float:
    # Hyperparameter suggestion
    params = {
        'zone_htg_setpoint_low': trial.suggest_float('zone_htg_setpoint_low', 19.5, 21.5),
        'zone_htg_setpoint_high': trial.suggest_float('zone_htg_setpoint_high', 19.0, 23.0),
        'zone_clg_setpoint_low': trial.suggest_float('zone_clg_setpoint_low', 22.0, 25.0),
        'zone_clg_setpoint_high': trial.suggest_float('zone_clg_setpoint_high', 22.5, 26.5),
        'sup_temp_at_low': trial.suggest_float('sup_temp_at_low', 17.0, 21.5),
        'sup_temp_at_high': trial.suggest_float('sup_temp_at_high', 16.0, 21.0),
        'co2_min_limit': trial.suggest_float('co2_min_limit', 500.0, 800.0),
        'co2_max_limit': trial.suggest_float('co2_max_limit', 700.0, 1000.0),
        'flow_low': trial.suggest_float('flow_low', 0.05, 0.40),
        'flow_moderate': trial.suggest_float('flow_moderate', 0.10, 0.70),
        'precondition_hours': trial.suggest_float('precondition_hours', 0.0, 3.0),
        'night_htg_setback': trial.suggest_float('night_htg_setback', 0.0, 1.0),
        'night_clg_setup': trial.suggest_float('night_clg_setup', 0.0, 1.0),
        'min_deadband': trial.suggest_float('min_deadband', 0.5, 2.0),
        'flow_night': trial.suggest_float('flow_night', 0.0, 0.20),
        'flow_pre_flush': trial.suggest_float('flow_pre_flush', 0.15, 0.80),
        'work_start': trial.suggest_float('work_start', 5.0, 7.5),
        'work_end': trial.suggest_float('work_end', 16.5, 20.0),
        'extended_end': trial.suggest_float('extended_end', 19.0, 24.0),
    }

    # Optional rules (ensure high > low)
    if params['zone_htg_setpoint_high'] < params['zone_htg_setpoint_low']:
        params['zone_htg_setpoint_high'] = params['zone_htg_setpoint_low']
    if params['zone_clg_setpoint_high'] < params['zone_clg_setpoint_low']:
        params['zone_clg_setpoint_high'] = params['zone_clg_setpoint_low']
    if params['flow_moderate'] < params['flow_low']:
        params['flow_moderate'] = params['flow_low']
    if params['flow_night'] > params['flow_low']:
        params['flow_night'] = params['flow_low']
    if params['flow_pre_flush'] < params['flow_low']:
        params['flow_pre_flush'] = params['flow_low']

    # For safety with EnergyPlus constraints, make sure cooling is always >= heating + 1.0
    if params['zone_clg_setpoint_low'] - params['zone_htg_setpoint_high'] < 1.0:
        params['zone_clg_setpoint_low'] = params['zone_htg_setpoint_high'] + 1.0
        
    if params['zone_clg_setpoint_high'] - params['zone_htg_setpoint_high'] < 1.0:
        params['zone_clg_setpoint_high'] = params['zone_htg_setpoint_high'] + 1.0
    if params['co2_max_limit'] <= params['co2_min_limit']:
        params['co2_max_limit'] = params['co2_min_limit'] + 1.0
    if params['work_end'] <= params['work_start'] + 4.0:
        params['work_end'] = params['work_start'] + 4.0
    if params['extended_end'] < params['work_end']:
        params['extended_end'] = params['work_end']

    costs, effective_params = run_trial_once(trial.number, params, keep_output=True)

    trial.set_user_attr("energy_cost", costs['energy_cost'])
    trial.set_user_attr("co2_penalty", costs['co2_penalty'])
    trial.set_user_attr("temp_penalty", costs['temp_penalty'])
    trial.set_user_attr("effective_params", effective_params)
    
    # Store final parameters and cost string
    print(
        f"Trial {trial.number} -> Total: {costs['total_cost']:.1f} "
        f"(E:{costs['energy_cost']:.1f}, CO2:{costs['co2_penalty']:.1f}, T:{costs['temp_penalty']:.1f})"
    )
    print(f"  Effective params: {effective_params}")

    global best_cost
    if costs['total_cost'] < best_cost:
        best_cost = costs['total_cost']
        best_dir = OPTUNA_OUT_DIR / "best_run"
        if best_dir.exists():
            shutil.rmtree(best_dir, ignore_errors=True)
        trial_dir = OPTUNA_OUT_DIR / f"trial_{trial.number}"
        if trial_dir.exists():
            shutil.copytree(trial_dir, best_dir)
        _save_best_snapshot(trial.number, params, effective_params, costs)
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
    else:
        trial_dir = OPTUNA_OUT_DIR / f"trial_{trial.number}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
        
    return costs['total_cost']

if __name__ == "__main__":
    args = parse_args()
    OPTUNA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    study_name = "vesa2_hvac_optimization"
    storage_name = f"sqlite:///{OPTUNA_OUT_DIR}/optuna_study.db"
    
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True, 
        direction="minimize"
    )
    ensure_context_matches(study, allow_mismatch=args.allow_context_mismatch)
    
    # Resume from existing best for proper best_run updates.
    if len(study.trials) > 0:
        try:
            best_cost = float(study.best_value)
        except Exception:
            best_cost = float("inf")
    else:
        best_cost = float("inf")
    n_trials = args.n_trials
    
    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
        pass

    print("\n" + "="*50)
    print("Optimization finished.")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Total Cost: {study.best_value:.2f} €")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.3f}")
    best_effective = study.best_trial.user_attrs.get("effective_params", {})
    if best_effective:
        print("Best Effective Parameters:")
        for key in sorted(best_effective.keys()):
            print(f"  {key}: {best_effective[key]:.6f}")
    print(f"Saved raw params: {BEST_RAW_JSON}")
    print(f"Saved effective params: {BEST_EFFECTIVE_JSON}")
    print(f"Saved best summary: {BEST_RESULT_TXT}")

    if args.verify_best:
        print("\nRe-running best trial once for verification...")
        verify_dir = OPTUNA_OUT_DIR / "verify_best_run"
        if verify_dir.exists():
            shutil.rmtree(verify_dir, ignore_errors=True)
        # Re-use logic but keep files this time for manual inspection.
        from pyenergyplus.api import EnergyPlusAPI
        from energyplus_controller import EnergyPlusController
        from parameterized_model import ParameterizedRBCModel

        model = ParameterizedRBCModel(**study.best_params)
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        controller = EnergyPlusController(api, model)
        api.runtime.callback_after_new_environment_warmup_complete(state, controller.initialize_handles)
        api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, controller.control_callback)
        verify_dir.mkdir(parents=True, exist_ok=True)
        eplus_args = ["-d", str(verify_dir), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]
        try:
            rc = api.runtime.run_energyplus(state, eplus_args)
        except TypeError:
            rc = api.runtime.run_energyplus(eplus_args)
        api.state_manager.delete_state(state)
        if int(rc) == 0 and (verify_dir / "eplusout.csv").exists():
            vdf = load_eplusout(str(verify_dir / "eplusout.csv"))
            vcosts = compute_total_cost(vdf)
            delta = float(vcosts["total_cost"] - study.best_value)
            print(
                f"Verification total: {vcosts['total_cost']:.3f} € "
                f"(study best: {study.best_value:.3f} €, delta: {delta:+.6f})"
            )
        else:
            print(f"Verification run failed (rc={int(rc)}). Inspect: {verify_dir}")
    print("="*50)
