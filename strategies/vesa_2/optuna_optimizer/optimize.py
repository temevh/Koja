import sys
import shutil
import optuna
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
STRATEGIES = SCRIPT_DIR.parent.parent
PROJECT_ROOT = STRATEGIES.parent
IDF_FILE = PROJECT_ROOT / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = PROJECT_ROOT / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
OPTUNA_OUT_DIR = SCRIPT_DIR / "optuna_out"

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

def load_eplusout(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Date/Time'] = df['Date/Time'].str.strip()
    mask_24 = df['Date/Time'].str.contains('24:00:00')
    df.loc[mask_24, 'Date/Time'] = df.loc[mask_24, 'Date/Time'].str.replace('24:00:00', '00:00:00')
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
        'zone_htg_setpoint_high': trial.suggest_float('zone_htg_setpoint_high', 19.5, 21.5), # Might be constrained if we want but trial limits are fine
        'zone_clg_setpoint_low': trial.suggest_float('zone_clg_setpoint_low', 23.0, 25.0),
        'zone_clg_setpoint_high': trial.suggest_float('zone_clg_setpoint_high', 24.0, 26.0),
        'sup_temp_at_low': trial.suggest_float('sup_temp_at_low', 17.0, 21.0),
        'sup_temp_at_high': trial.suggest_float('sup_temp_at_high', 17.0, 21.0),
        'co2_min_limit': trial.suggest_float('co2_min_limit', 500.0, 750.0),
        'co2_max_limit': trial.suggest_float('co2_max_limit', 750.0, 950.0),
        'flow_low': trial.suggest_float('flow_low', 0.10, 0.35),
        'flow_moderate': trial.suggest_float('flow_moderate', 0.20, 0.50),
    }

    # Optional rules (ensure high > low)
    if params['zone_htg_setpoint_high'] < params['zone_htg_setpoint_low']:
        params['zone_htg_setpoint_high'] = params['zone_htg_setpoint_low']
    if params['zone_clg_setpoint_high'] < params['zone_clg_setpoint_low']:
        params['zone_clg_setpoint_high'] = params['zone_clg_setpoint_low']
    if params['flow_moderate'] < params['flow_low']:
        params['flow_moderate'] = params['flow_low']

    # For safety with EnergyPlus constraints, make sure cooling is always >= heating + 1.0
    if params['zone_clg_setpoint_low'] - params['zone_htg_setpoint_high'] < 1.0:
        params['zone_clg_setpoint_low'] = params['zone_htg_setpoint_high'] + 1.0
        
    if params['zone_clg_setpoint_high'] - params['zone_htg_setpoint_high'] < 1.0:
        params['zone_clg_setpoint_high'] = params['zone_htg_setpoint_high'] + 1.0
    if params['co2_max_limit'] <= params['co2_min_limit']:
        params['co2_max_limit'] = params['co2_min_limit'] + 1.0

    from pyenergyplus.api import EnergyPlusAPI
    from energyplus_controller import EnergyPlusController
    from parameterized_model import ParameterizedRBCModel
    
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    model = ParameterizedRBCModel(**params)
    effective_params = {
        'zone_htg_setpoint_low': model.ZONE_HTG_SETPOINT_LOW,
        'zone_htg_setpoint_high': model.ZONE_HTG_SETPOINT_HIGH,
        'zone_clg_setpoint_low': model.ZONE_CLG_SETPOINT_LOW,
        'zone_clg_setpoint_high': model.ZONE_CLG_SETPOINT_HIGH,
        'sup_temp_at_low': model.SUP_TEMP_AT_LOW,
        'sup_temp_at_high': model.SUP_TEMP_AT_HIGH,
        'co2_min_limit': model.CO2_MIN_LIMIT,
        'co2_max_limit': model.CO2_MAX_LIMIT,
        'flow_low': model.FLOW_LOW,
        'flow_moderate': model.FLOW_MODERATE,
    }
    controller = EnergyPlusController(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(state, controller.initialize_handles)
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, controller.control_callback)

    trial_dir = OPTUNA_OUT_DIR / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    args = ["-d", str(trial_dir), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]
    
    try:
        rc = api.runtime.run_energyplus(state, args)
    except TypeError:
        rc = api.runtime.run_energyplus(args)

    api.state_manager.delete_state(state)

    if rc != 0:
        print(f"Trial {trial.number} failed with RC={rc}")
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
        shutil.copytree(trial_dir, best_dir)

    # Cleanup large files to save disk usage
    if trial_dir.exists():
        shutil.rmtree(trial_dir, ignore_errors=True)
        
    return costs['total_cost']

if __name__ == "__main__":
    OPTUNA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    study_name = "vesa2_hvac_optimization"
    storage_name = f"sqlite:///{OPTUNA_OUT_DIR}/optuna_study.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True, 
        direction="minimize"
    )
    
    # Run 100 trials, can be stopped and resumed
    n_trials = 100
    
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
    print("="*50)
