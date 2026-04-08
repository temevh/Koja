"""Optuna Bayesian optimization for vesa_4 model parameters.

Runs EnergyPlus simulations with different parameter sets and minimizes
total cost (energy + CO2 penalty + temperature penalty).

Usage:
    cd strategies/vesa_4
    python optimize.py

Each trial runs a full-year simulation (~20s). Ctrl+C to stop early;
the best result is preserved and progress is stored in an SQLite DB.
"""

import sys
import shutil
import time
from pathlib import Path
from typing import Dict

import optuna
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
STRATEGIES = SCRIPT_DIR.parent
PROJECT_ROOT = STRATEGIES.parent
IDF_FILE = PROJECT_ROOT / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = PROJECT_ROOT / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
OPTUNA_OUT = SCRIPT_DIR / "optuna_out"

ELEC_PRICE = 0.11
GAS_PRICE = 0.06
J_TO_KWH = 1 / 3.6e6

CO2_P1, CO2_P2, CO2_P3 = 2, 10, 50
TEMP_P1, TEMP_P2, TEMP_P3 = 1, 5, 25

COLUMN_NAMES = [
    'Time', 'Outdoor_Tdb_C', 'Outdoor_Twb_C',
    'Space1_occupants', 'Space2_occupants', 'Space3_occupants',
    'Space4_occupants', 'Space5_occupants',
    'lights-1', 'lights-2', 'lights-3', 'lights-4', 'lights-5',
    'equip-1', 'equip-2', 'equip-3', 'equip-4', 'equip-5',
    'Plenum1_T_C', 'Plenum1_RH_%',
    'Space1_T_C', 'Space1_RH_%', 'Space2_T_C', 'Space2_RH_%',
    'Space3_T_C', 'Space3_RH_%', 'Space4_T_C', 'Space4_RH_%',
    'Space5_T_C', 'Space5_RH_%',
    'Plenum_CO2_ppm', 'Plenum_CO2_pred', 'Plenum_CO2_setpoint_ppm',
    'Plenum_CO2_internal_gain',
    'Space1_CO2_ppm', 'Space1_CO2_pred', 'Space1_CO2_setpoint_ppm',
    'Space1_CO2_internal_gain',
    'Space2_CO2_ppm', 'Space2_CO2_pred', 'Space2_CO2_setpoint_ppm',
    'Space2_CO2_internal_gain',
    'Space3_CO2_ppm', 'Space3_CO2_pred', 'Space3_CO2_setpoint_ppm',
    'Space3_CO2_internal_gain',
    'Space4_CO2_ppm', 'Space4_CO2_pred', 'Space4_CO2_setpoint_ppm',
    'Space4_CO2_internal_gain',
    'Space5_CO2_ppm', 'Space5_CO2_pred', 'Space5_CO2_setpoint_ppm',
    'Space5_CO2_internal_gain',
    'doas_fan', 'fcu_1', 'fcu_2', 'fcu_3', 'fcu_4', 'fcu_5',
    'hex', 'chiller', 'tower', 'boiler', 'coldw_pump', 'condw_pump',
    'hotw_pump',
    'Node2_T_C', 'Node2_Mdot_kg/s', 'Node2_W_Ratio',
    'Node2_SP_T_C', 'Node2_CO2_ppm', 'Node1_T_C',
    'Gas_Facility_E_J', 'Elec_Facility_E_J', 'Elec_HVAC_E_J',
    'CoolingCoils:EnergyTransfer', 'HeatingCoils:EnergyTransfer',
    'ElectricityNet:Facility',
    'General:Cooling:EnergyTransfer', 'Cooling:EnergyTransfer',
]

ZONE_CO2_COLS = [f'Space{i}_CO2_ppm' for i in range(1, 6)]
best_cost = float('inf')


def load_eplusout(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Date/Time'] = df['Date/Time'].str.strip()
    mask_24 = df['Date/Time'].str.contains('24:00:00')
    df.loc[mask_24, 'Date/Time'] = df.loc[mask_24, 'Date/Time'].str.replace(
        '24:00:00', '00:00:00'
    )
    df['Date/Time'] = '2024/' + df['Date/Time']
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%Y/%m/%d %H:%M:%S')
    df.loc[mask_24, 'Date/Time'] += pd.Timedelta(days=1)
    df.columns = COLUMN_NAMES
    return df


def compute_energy_cost(df: pd.DataFrame) -> float:
    elec_kwh = df['Elec_Facility_E_J'].sum() * J_TO_KWH
    gas_kwh = df['Gas_Facility_E_J'].sum() * J_TO_KWH
    return elec_kwh * ELEC_PRICE + gas_kwh * GAS_PRICE


def compute_temperature_limits(df: pd.DataFrame):
    t_mean = df['Outdoor_Tdb_C'].rolling(24, min_periods=24).mean()
    t = pd.to_numeric(t_mean, errors='coerce').to_numpy(dtype=float)
    lower_s1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_s1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.5 + 0.166 * t, 25.0))
    lower_s2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_s2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20 * t, 26.0))
    lower_s3 = np.full_like(t, 20.0)
    upper_s3 = np.where(t <= 10, 25.0, 27.0)
    return lower_s1, upper_s1, lower_s2, upper_s2, lower_s3, upper_s3


def compute_co2_penalty(df: pd.DataFrame) -> float:
    total = 0.0
    for col in ZONE_CO2_COLS:
        co2 = df[col]
        total += CO2_P1 * ((co2 > 770) & (co2 <= 970)).sum()
        total += CO2_P2 * ((co2 > 970) & (co2 <= 1220)).sum()
        total += CO2_P3 * (co2 > 1220).sum()
    return total


def compute_temperature_penalty(df: pd.DataFrame) -> float:
    limits = compute_temperature_limits(df)
    lower_s1, upper_s1, lower_s2, upper_s2, lower_s3, upper_s3 = limits
    total = 0.0
    for i in range(1, 6):
        t_in = pd.to_numeric(df[f'Space{i}_T_C'], errors='coerce').to_numpy(dtype=float)
        in_s1 = (t_in >= lower_s1) & (t_in <= upper_s1)
        in_s2 = (t_in >= lower_s2) & (t_in <= upper_s2)
        in_s3 = (t_in >= lower_s3) & (t_in <= upper_s3)
        total += TEMP_P1 * (~in_s1 & in_s2).sum()
        total += TEMP_P2 * (~in_s2 & in_s3).sum()
        total += TEMP_P3 * (~in_s3).sum()
    return total


def compute_total_cost(df: pd.DataFrame) -> Dict[str, float]:
    energy = compute_energy_cost(df)
    co2 = compute_co2_penalty(df)
    temp = compute_temperature_penalty(df)
    return {
        'energy_cost': energy, 'co2_penalty': co2,
        'temp_penalty': temp, 'total_cost': energy + co2 + temp,
    }


def objective(trial: optuna.Trial) -> float:
    global best_cost

    params = {
        'htg_margin': trial.suggest_float('htg_margin', 0.1, 0.8),
        'clg_margin': trial.suggest_float('clg_margin', 0.1, 0.8),
        'night_htg_setback': trial.suggest_float('night_htg_setback', 0.3, 1.5),
        'night_clg_setup': trial.suggest_float('night_clg_setup', 0.3, 1.5),
        'precondition_hours': trial.suggest_float('precondition_hours', 0.5, 2.0),
        'sup_temp_cold': trial.suggest_float('sup_temp_cold', 17.0, 21.0),
        'sup_temp_warm': trial.suggest_float('sup_temp_warm', 16.0, 19.0),
        'co2_low_threshold': trial.suggest_float('co2_low_threshold', 500.0, 700.0),
        'co2_high_threshold': trial.suggest_float('co2_high_threshold', 700.0, 900.0),
        'flow_min_occupied': trial.suggest_float('flow_min_occupied', 0.08, 0.25),
        'flow_moderate': trial.suggest_float('flow_moderate', 0.25, 0.50),
        'flow_pre_flush': trial.suggest_float('flow_pre_flush', 0.25, 0.60),
        'work_start': trial.suggest_float('work_start', 5.0, 7.0),
        'work_end': trial.suggest_float('work_end', 17.0, 19.0),
        'min_deadband': trial.suggest_float('min_deadband', 0.5, 2.0),
    }

    if params['co2_high_threshold'] <= params['co2_low_threshold'] + 50:
        params['co2_high_threshold'] = params['co2_low_threshold'] + 100

    from pyenergyplus.api import EnergyPlusAPI
    from energyplus_controller import EnergyPlusController
    from my_model import Vesa4Model

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    model = Vesa4Model(**params)
    controller = EnergyPlusController(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(
        state, controller.initialize_handles
    )
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        state, controller.control_callback
    )

    trial_dir = OPTUNA_OUT / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    args = ["-d", str(trial_dir), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]

    t0 = time.time()
    try:
        rc = api.runtime.run_energyplus(state, args)
    except TypeError:
        rc = api.runtime.run_energyplus(args)
    elapsed = time.time() - t0

    api.state_manager.delete_state(state)

    if rc != 0:
        print(f"Trial {trial.number} FAILED (rc={rc}, {elapsed:.1f}s)")
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
        raise optuna.TrialPruned()

    csv_path = trial_dir / "eplusout.csv"
    if not csv_path.exists():
        raise optuna.TrialPruned()

    df = load_eplusout(str(csv_path))
    costs = compute_total_cost(df)

    trial.set_user_attr("energy_cost", costs['energy_cost'])
    trial.set_user_attr("co2_penalty", costs['co2_penalty'])
    trial.set_user_attr("temp_penalty", costs['temp_penalty'])
    trial.set_user_attr("elapsed_s", elapsed)

    print(
        f"Trial {trial.number} -> Total: {costs['total_cost']:.1f}€ "
        f"(E:{costs['energy_cost']:.1f}, CO2:{costs['co2_penalty']:.1f}, "
        f"T:{costs['temp_penalty']:.1f}) [{elapsed:.1f}s]"
    )

    if costs['total_cost'] < best_cost:
        best_cost = costs['total_cost']
        best_dir = OPTUNA_OUT / "best_run"
        if best_dir.exists():
            shutil.rmtree(best_dir, ignore_errors=True)
        shutil.copytree(trial_dir, best_dir)
        print(f"  -> NEW BEST: {best_cost:.1f}€")

    if trial_dir.exists():
        shutil.rmtree(trial_dir, ignore_errors=True)

    return costs['total_cost']


if __name__ == "__main__":
    OPTUNA_OUT.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="vesa4_hvac_optimization",
        storage=f"sqlite:///{OPTUNA_OUT}/optuna_study.db",
        load_if_exists=True,
        direction="minimize",
    )

    n_trials = 100
    print(f"Starting optimization ({n_trials} trials, ~{n_trials * 20 // 60} min)")

    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\n" + "=" * 60)
    print("Optimization finished.")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Total Cost: {study.best_value:.2f} €")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60)

    print("\nTo run with best parameters, update Vesa4Model() in run_idf.py:")
    print("  model = Vesa4Model(")
    for key, value in study.best_params.items():
        print(f"      {key}={value:.4f},")
    print("  )")
