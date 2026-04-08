#!/usr/bin/env python3
"""Automated parameter sweep for RBC model optimisation.

Runs multiple EnergyPlus simulations with different parameter combinations
and calculates the total cost (energy + CO2 penalty + temperature penalty)
for each. Results are saved to a CSV file sorted by total cost.

Usage:
    cd rbc_full_on
    python sweep_params.py

Each simulation takes ~20 seconds, so plan accordingly.
The script prints progress and a summary table at the end.
"""

import sys
import os
import shutil
import itertools
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# ── EnergyPlus setup ────────────────────────────────────────────────────────

ENERGYPLUS_DIR = r"C:\EnergyPlusV25-2-0"
sys.path.append(ENERGYPLUS_DIR)

# ── Paths — resolved from script location so it works from any CWD ──────────
SCRIPT_DIR   = Path(__file__).resolve().parent               # strategies/teemu/
STRATEGIES   = SCRIPT_DIR.parent                             # strategies/
PROJECT_ROOT = STRATEGIES.parent                             # Koja/

IDF_FILE = PROJECT_ROOT / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = PROJECT_ROOT / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"
BASE_OUT_DIR = SCRIPT_DIR / "sweep_output"

# Source for the energyplus_controller.py (shared across strategies)
CONTROLLER_SRC = STRATEGIES / "rbc_full_on" / "energyplus_controller.py"


# ── Cost constants (must match the scoring notebook) ────────────────────────

ELEC_PRICE = 0.11   # €/kWh
GAS_PRICE  = 0.06   # €/kWh
J_TO_KWH   = 1 / 3.6e6

CO2_P1 = 2    # €/h  >770 ppm but ≤970 ppm
CO2_P2 = 10   # €/h  >970 ppm but ≤1220 ppm
CO2_P3 = 50   # €/h  >1220 ppm

TEMP_P1 = 1   # €/h  outside S1 but inside S2
TEMP_P2 = 5   # €/h  outside S2 but inside S3
TEMP_P3 = 25  # €/h  outside S3

# ── Column names for the EnergyPlus output CSV ──────────────────────────────

COLUMN_NAMES = [
    'Time',
    'Outdoor_Tdb_C',
    'Outdoor_Twb_C',
    'Space1_occupants', 'Space2_occupants', 'Space3_occupants',
    'Space4_occupants', 'Space5_occupants',
    'lights-1', 'lights-2', 'lights-3', 'lights-4', 'lights-5',
    'equip-1', 'equip-2', 'equip-3', 'equip-4', 'equip-5',
    'Plenum1_T_C', 'Plenum1_RH_%',
    'Space1_T_C', 'Space1_RH_%',
    'Space2_T_C', 'Space2_RH_%',
    'Space3_T_C', 'Space3_RH_%',
    'Space4_T_C', 'Space4_RH_%',
    'Space5_T_C', 'Space5_RH_%',
    'Plenum_CO2_ppm', 'Plenum_CO2_pred', 'Plenum_CO2_setpoint_ppm', 'Plenum_CO2_internal_gain',
    'Space1_CO2_ppm', 'Space1_CO2_pred', 'Space1_CO2_setpoint_ppm', 'Space1_CO2_internal_gain',
    'Space2_CO2_ppm', 'Space2_CO2_pred', 'Space2_CO2_setpoint_ppm', 'Space2_CO2_internal_gain',
    'Space3_CO2_ppm', 'Space3_CO2_pred', 'Space3_CO2_setpoint_ppm', 'Space3_CO2_internal_gain',
    'Space4_CO2_ppm', 'Space4_CO2_pred', 'Space4_CO2_setpoint_ppm', 'Space4_CO2_internal_gain',
    'Space5_CO2_ppm', 'Space5_CO2_pred', 'Space5_CO2_setpoint_ppm', 'Space5_CO2_internal_gain',
    'doas_fan',
    'fcu_1', 'fcu_2', 'fcu_3', 'fcu_4', 'fcu_5',
    'hex', 'chiller', 'tower', 'boiler',
    'coldw_pump', 'condw_pump', 'hotw_pump',
    'Node2_T_C', 'Node2_Mdot_kg/s', 'Node2_W_Ratio',
    'Node2_SP_T_C', 'Node2_CO2_ppm', 'Node1_T_C',
    'Gas_Facility_E_J', 'Elec_Facility_E_J', 'Elec_HVAC_E_J',
    'CoolingCoils:EnergyTransfer', 'HeatingCoils:EnergyTransfer',
    'ElectricityNet:Facility',
    'General:Cooling:EnergyTransfer', 'Cooling:EnergyTransfer',
]

ZONE_CO2_COLS = [f'Space{i}_CO2_ppm' for i in range(1, 6)]


# ===========================================================================
# PARAMETER GRID — edit this to define what combinations to test
# ===========================================================================
# Each key is a parameter name, each value is a list of values to try.
# The script will test ALL combinations (cartesian product).
#
# ⚠ Be careful with the number of combinations!
#   3 params × 3 values each = 27 runs × ~20s = ~9 minutes
#   4 params × 3 values each = 81 runs × ~20s = ~27 minutes

PARAM_GRID = {
    # Zone heating setpoint at outdoor <= 0°C  [°C]
    "ZONE_HTG_SETPOINT_LOW":  [21.0, 21.5],

    # Zone cooling setpoint at outdoor <= 0°C  [°C]
    "ZONE_CLG_SETPOINT_LOW":  [21.5, 22.0],

    # Supply air temp when return air is cool  [°C]
    "SUP_TEMP_AT_LOW":        [19.0, 20.0, 21.0],

    # Supply air temp when return air is warm  [°C]
    "SUP_TEMP_AT_HIGH":       [18.0, 19.0, 20.0],
}

# These parameters stay fixed for all runs (not swept):
FIXED_PARAMS = {
    "ZONE_HTG_SETPOINT_HIGH": 22.5,
    "ZONE_CLG_SETPOINT_HIGH": 24.5,
    "OUTDOOR_TEMP_LOW":       0.0,
    "OUTDOOR_TEMP_HIGH":      20.0,
    "RETURN_AIR_TEMP_LOW":    21.5,
    "RETURN_AIR_TEMP_HIGH":   24.5,
    "CO2_MIN_LIMIT":          500,
    "CO2_MAX_LIMIT":          750,
    "OUTDOOR_TEMP_COLD_LIMIT": -25.0,
    "OUTDOOR_TEMP_COLD_UPPER": -15.0,
    "FLOW_LOW":               0.20,
    "FLOW_MODERATE":          0.40,
    "FLOW_BOOST":             1.0,
    "FLOW_MAX":               1.0,
}


# ===========================================================================
# Cost calculation (replicates notebook logic exactly)
# ===========================================================================

def load_eplusout(csv_path: str) -> pd.DataFrame:
    """Load an EnergyPlus CSV, fix the 24:00 timestamp, rename columns."""
    df = pd.read_csv(csv_path)
    df['Date/Time'] = df['Date/Time'].str.strip()

    mask_24 = df['Date/Time'].str.contains('24:00:00')
    df.loc[mask_24, 'Date/Time'] = (
        df.loc[mask_24, 'Date/Time'].str.replace('24:00:00', '00:00:00')
    )
    df['Date/Time'] = '2024/' + df['Date/Time']
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%Y/%m/%d %H:%M:%S')
    df.loc[mask_24, 'Date/Time'] += pd.Timedelta(days=1)

    df.columns = COLUMN_NAMES
    return df


def compute_energy_cost(df: pd.DataFrame) -> float:
    """Compute total annual energy cost in €."""
    elec_kwh = df['Elec_Facility_E_J'].sum() * J_TO_KWH
    gas_kwh  = df['Gas_Facility_E_J'].sum() * J_TO_KWH
    return elec_kwh * ELEC_PRICE + gas_kwh * GAS_PRICE


def compute_temperature_limits(df: pd.DataFrame):
    """Compute S1/S2/S3 temperature limits from 24h rolling outdoor temp."""
    t_mean = df['Outdoor_Tdb_C'].rolling(24, min_periods=24).mean()
    t = pd.to_numeric(t_mean, errors='coerce').to_numpy(dtype=float)

    lower_S1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_S1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.5 + 0.166 * t, 25.0))
    lower_S2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_S2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20  * t, 26.0))
    lower_S3 = np.full_like(t, 20.0)
    upper_S3 = np.where(t <= 10, 25.0, 27.0)

    return lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3


def compute_co2_penalty(df: pd.DataFrame) -> float:
    """Compute total CO2 penalty in €."""
    total = 0.0
    for col in ZONE_CO2_COLS:
        co2 = df[col]
        h_s1_s2 = ((co2 > 770) & (co2 <= 970)).sum()
        h_s2_s3 = ((co2 > 970) & (co2 <= 1220)).sum()
        h_above  = (co2 > 1220).sum()
        total += CO2_P1 * h_s1_s2 + CO2_P2 * h_s2_s3 + CO2_P3 * h_above
    return total


def compute_temperature_penalty(df: pd.DataFrame) -> float:
    """Compute total temperature penalty in €."""
    limits = compute_temperature_limits(df)
    lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3 = limits

    total = 0.0
    for i in range(1, 6):
        col = f'Space{i}_T_C'
        t_in = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)

        in_s1 = (t_in >= lower_S1) & (t_in <= upper_S1)
        in_s2 = (t_in >= lower_S2) & (t_in <= upper_S2)
        in_s3 = (t_in >= lower_S3) & (t_in <= upper_S3)

        h_s1_only = (~in_s1 & in_s2).sum()
        h_s2_only = (~in_s2 & in_s3).sum()
        h_out     = (~in_s3).sum()

        total += TEMP_P1 * h_s1_only + TEMP_P2 * h_s2_only + TEMP_P3 * h_out
    return total


def compute_total_cost(df: pd.DataFrame) -> Dict[str, float]:
    """Compute all cost components and return as dict."""
    energy = compute_energy_cost(df)
    co2    = compute_co2_penalty(df)
    temp   = compute_temperature_penalty(df)
    return {
        'energy_cost': energy,
        'co2_penalty': co2,
        'temp_penalty': temp,
        'total_cost': energy + co2 + temp,
    }


# ===========================================================================
# Simulation runner
# ===========================================================================

def write_model_file(params: Dict, filepath: Path) -> None:
    """Generate an rbc_model_1.py with the given parameter values."""
    p = {**FIXED_PARAMS, **params}

    code = f'''"""Auto-generated RBC model for parameter sweep."""

from dataclasses import dataclass
from typing import Tuple

# ── Parameters (auto-generated by sweep_params.py) ──
ZONE_HTG_SETPOINT_LOW  = {p["ZONE_HTG_SETPOINT_LOW"]}
ZONE_HTG_SETPOINT_HIGH = {p["ZONE_HTG_SETPOINT_HIGH"]}
ZONE_CLG_SETPOINT_LOW  = {p["ZONE_CLG_SETPOINT_LOW"]}
ZONE_CLG_SETPOINT_HIGH = {p["ZONE_CLG_SETPOINT_HIGH"]}

OUTDOOR_TEMP_LOW  = {p["OUTDOOR_TEMP_LOW"]}
OUTDOOR_TEMP_HIGH = {p["OUTDOOR_TEMP_HIGH"]}

RETURN_AIR_TEMP_LOW  = {p["RETURN_AIR_TEMP_LOW"]}
RETURN_AIR_TEMP_HIGH = {p["RETURN_AIR_TEMP_HIGH"]}
SUP_TEMP_AT_LOW  = {p["SUP_TEMP_AT_LOW"]}
SUP_TEMP_AT_HIGH = {p["SUP_TEMP_AT_HIGH"]}

CO2_MIN_LIMIT = {p["CO2_MIN_LIMIT"]}
CO2_MAX_LIMIT = {p["CO2_MAX_LIMIT"]}

OUTDOOR_TEMP_COLD_LIMIT = {p["OUTDOOR_TEMP_COLD_LIMIT"]}
OUTDOOR_TEMP_COLD_UPPER = {p["OUTDOOR_TEMP_COLD_UPPER"]}

FLOW_LOW      = {p["FLOW_LOW"]}
FLOW_MODERATE = {p["FLOW_MODERATE"]}
FLOW_BOOST    = {p["FLOW_BOOST"]}
FLOW_MAX      = {p["FLOW_MAX"]}


@dataclass
class AHUModeParams:
    flow: float
    mode_name: str


class RBCModel:
    def __init__(self) -> None:
        self.htg_setpoint = ZONE_HTG_SETPOINT_LOW
        self.clg_setpoint = ZONE_CLG_SETPOINT_LOW
        self._co2_boosting = False
        self.mode_params = {{
            0: AHUModeParams(flow=0.0,          mode_name="off"),
            1: AHUModeParams(flow=FLOW_LOW,      mode_name="low"),
            2: AHUModeParams(flow=FLOW_MODERATE,  mode_name="moderate"),
            3: AHUModeParams(flow=FLOW_BOOST,     mode_name="boost"),
        }}

    def zone_setpoints(self, outdoor_temp: float) -> Tuple[float, float]:
        t = max(0.0, min(1.0,
            (outdoor_temp - OUTDOOR_TEMP_LOW) / (OUTDOOR_TEMP_HIGH - OUTDOOR_TEMP_LOW)
        ))
        htg = ZONE_HTG_SETPOINT_LOW + t * (ZONE_HTG_SETPOINT_HIGH - ZONE_HTG_SETPOINT_LOW)
        clg = ZONE_CLG_SETPOINT_LOW + t * (ZONE_CLG_SETPOINT_HIGH - ZONE_CLG_SETPOINT_LOW)
        return htg, clg

    def return_air_compensation(self, return_air_temp: float) -> float:
        if return_air_temp <= RETURN_AIR_TEMP_LOW:
            return SUP_TEMP_AT_LOW
        if return_air_temp >= RETURN_AIR_TEMP_HIGH:
            return SUP_TEMP_AT_HIGH
        slope = (SUP_TEMP_AT_HIGH - SUP_TEMP_AT_LOW) / (RETURN_AIR_TEMP_HIGH - RETURN_AIR_TEMP_LOW)
        result = SUP_TEMP_AT_LOW + slope * (return_air_temp - RETURN_AIR_TEMP_LOW)
        return max(SUP_TEMP_AT_HIGH, min(SUP_TEMP_AT_LOW, result))

    def co2_flow_control(self, hour, day, co2_concentration, outdoor_temp):
        is_workday = day not in (1, 7)
        is_pre_flush       = is_workday and 5.5 <= hour < 6.5
        is_weekend_daytime = (not is_workday) and 8.0 <= hour < 16.0
        is_working_hours   = is_workday and 6.5 <= hour < 19.0

        if is_pre_flush:
            base_flow = FLOW_MODERATE
        elif is_weekend_daytime or is_working_hours:
            base_flow = FLOW_LOW
        else:
            base_flow = 0.0

        if co2_concentration <= CO2_MIN_LIMIT:
            co2_flow = 0.0
        elif co2_concentration >= CO2_MAX_LIMIT:
            co2_flow = FLOW_BOOST
        else:
            fraction = (co2_concentration - CO2_MIN_LIMIT) / (CO2_MAX_LIMIT - CO2_MIN_LIMIT)
            co2_flow = FLOW_LOW + fraction * (FLOW_BOOST - FLOW_LOW)

        target_flow = max(co2_flow, base_flow)

        if outdoor_temp <= OUTDOOR_TEMP_COLD_LIMIT:
            max_allowed = FLOW_LOW
        elif outdoor_temp >= OUTDOOR_TEMP_COLD_UPPER:
            max_allowed = FLOW_MAX
        else:
            slope = (FLOW_MAX - FLOW_LOW) / (OUTDOOR_TEMP_COLD_UPPER - OUTDOOR_TEMP_COLD_LIMIT)
            max_allowed = FLOW_LOW + slope * (outdoor_temp - OUTDOOR_TEMP_COLD_LIMIT)

        return min(target_flow, max_allowed)

    def calculate_setpoints(self, zone_temp, outdoor_temp, return_air_temp,
                            occupancy, hour, day, co2_concentration):
        supply_air_temp = self.return_air_compensation(return_air_temp)
        htg, clg = self.zone_setpoints(outdoor_temp)
        self.htg_setpoint = htg
        self.clg_setpoint = clg
        flow = self.co2_flow_control(hour, day, co2_concentration, outdoor_temp)
        return self.htg_setpoint, self.clg_setpoint, supply_air_temp, flow
'''
    filepath.write_text(code, encoding='utf-8')


# Shared temp directory — reused for every run, wiped in between
TEMP_RUN_DIR = BASE_OUT_DIR / "_temp_run"
BEST_RUN_DIR = BASE_OUT_DIR / "best_run"


def run_simulation(params: Dict, run_id: int, total: int) -> Dict:
    """Run one EnergyPlus simulation with the given params and return costs."""
    from pyenergyplus.api import EnergyPlusAPI

    # Wipe and recreate the shared temp directory
    if TEMP_RUN_DIR.exists():
        shutil.rmtree(TEMP_RUN_DIR)
    TEMP_RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Write the model file into the temp dir so we can import it
    model_file = TEMP_RUN_DIR / "rbc_model_1.py"
    write_model_file(params, model_file)

    # Copy the energyplus_controller from the rbc_full_on strategy
    if CONTROLLER_SRC.exists():
        shutil.copy2(CONTROLLER_SRC, TEMP_RUN_DIR / "energyplus_controller.py")
    else:
        raise FileNotFoundError(f"Controller not found: {CONTROLLER_SRC}")

    # Add temp dir (for rbc_model_1) and project root (for shared.variable_config) to path
    sys.path.insert(0, str(TEMP_RUN_DIR))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Force reimport of the module (since module name is always the same)
    if 'rbc_model_1' in sys.modules:
        del sys.modules['rbc_model_1']
    if 'energyplus_controller' in sys.modules:
        del sys.modules['energyplus_controller']

    from rbc_model_1 import RBCModel as RunModel
    from energyplus_controller import EnergyPlusController

    sys.path.remove(str(TEMP_RUN_DIR))

    # --- Run EnergyPlus ---
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()

    model = RunModel()
    controller = EnergyPlusController(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(
        state, controller.initialize_handles
    )
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
        state, controller.control_callback
    )

    eplus_out = TEMP_RUN_DIR / "eplus_out"
    eplus_out.mkdir(parents=True, exist_ok=True)

    args = ["-d", str(eplus_out), "-w", str(EPW_FILE), "-r", str(IDF_FILE)]

    t0 = time.time()
    try:
        rc = api.runtime.run_energyplus(state, args)
    except TypeError:
        rc = api.runtime.run_energyplus(args)
    elapsed = time.time() - t0

    # Clean up the state
    api.state_manager.delete_state(state)

    if rc != 0:
        print(f"  ✗ Run {run_id} FAILED (rc={rc}) — skipping")
        return {**params, 'run_id': run_id, 'rc': rc,
                'energy_cost': float('nan'), 'co2_penalty': float('nan'),
                'temp_penalty': float('nan'), 'total_cost': float('nan'),
                'elapsed_s': elapsed}

    # --- Calculate costs ---
    csv_path = eplus_out / "eplusout.csv"
    df = load_eplusout(str(csv_path))
    costs = compute_total_cost(df)

    result = {**params, 'run_id': run_id, 'rc': rc, **costs, 'elapsed_s': elapsed}
    return result


def save_best_run(params: Dict, total_cost: float) -> None:
    """Copy the current temp run output to the best_run folder."""
    if BEST_RUN_DIR.exists():
        shutil.rmtree(BEST_RUN_DIR)
    shutil.copytree(TEMP_RUN_DIR, BEST_RUN_DIR)

    # Write a summary file so you know which params produced this
    summary = BEST_RUN_DIR / "best_params.txt"
    lines = [f"Total cost: {total_cost:.2f} €\n", "\nParameters:\n"]
    for k, v in params.items():
        lines.append(f"  {k} = {v}\n")
    summary.write_text("".join(lines), encoding='utf-8')


# ===========================================================================
# Main sweep
# ===========================================================================

def main():
    # Build all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    combos = list(itertools.product(*param_values))
    total = len(combos)

    print("=" * 70)
    print(f"  PARAMETER SWEEP — {total} combinations to test")
    print("=" * 70)
    print(f"  Swept parameters: {param_names}")
    print(f"  Estimated time:   ~{total * 20 // 60} min ({total} × ~20s)")
    print("=" * 70)
    print()

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    best_cost = float('inf')

    for i, values in enumerate(combos):
        params = dict(zip(param_names, values))

        # Pretty-print current params
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{i+1}/{total}] {param_str}")

        result = run_simulation(params, run_id=i, total=total)
        results.append(result)

        if result['rc'] == 0:
            is_new_best = result['total_cost'] < best_cost
            marker = "  ★ NEW BEST!" if is_new_best else ""
            print(f"  ✓ Energy={result['energy_cost']:.0f}€  "
                  f"CO2={result['co2_penalty']:.0f}€  "
                  f"Temp={result['temp_penalty']:.0f}€  "
                  f"TOTAL={result['total_cost']:.0f}€  "
                  f"({result['elapsed_s']:.1f}s){marker}")

            if is_new_best:
                best_cost = result['total_cost']
                save_best_run(params, best_cost)
                print(f"  → Saved to {BEST_RUN_DIR}/")
        print()

    # Clean up temp directory after sweep is done
    if TEMP_RUN_DIR.exists():
        shutil.rmtree(TEMP_RUN_DIR)

    # --- Save & display results ---
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('total_cost')

    csv_out = BASE_OUT_DIR / "sweep_results.csv"
    df_results.to_csv(csv_out, index=False)

    print()
    print("=" * 70)
    print("  SWEEP COMPLETE — Results sorted by total cost")
    print("=" * 70)

    # Display top 10
    display_cols = param_names + ['energy_cost', 'co2_penalty', 'temp_penalty', 'total_cost']
    top = df_results[display_cols].head(10)
    print(top.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print()
    print(f"Full results saved to: {csv_out.resolve()}")

    # Highlight the best
    best = df_results.iloc[0]
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  BEST CONFIGURATION                                 ║")
    print("╠══════════════════════════════════════════════════════╣")
    for p in param_names:
        print(f"║  {p:<35s} = {best[p]:>8.2f}  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Energy cost:          {best['energy_cost']:>10.2f} €            ║")
    print(f"║  CO2 penalty:          {best['co2_penalty']:>10.2f} €            ║")
    print(f"║  Temperature penalty:  {best['temp_penalty']:>10.2f} €            ║")
    print(f"║  TOTAL:                {best['total_cost']:>10.2f} €            ║")
    print("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
