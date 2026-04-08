"""Optuna optimization for vesa_6 with auto-persisted best parameters."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
IDF_FILE = PROJECT_ROOT / "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf"
EPW_FILE = PROJECT_ROOT / "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw"

OPTUNA_OUT = SCRIPT_DIR / "optuna_out"
BEST_RUN_DIR = OPTUNA_OUT / "best_run"
BEST_JSON = SCRIPT_DIR / "best_params.json"
BEST_PY = SCRIPT_DIR / "best_params.py"
BEST_TXT = SCRIPT_DIR / "best_result.txt"
TARGET_TOTAL = float(os.environ.get("VESA6_TARGET", "7000"))

ELEC_PRICE = 0.11
GAS_PRICE = 0.06
J_TO_KWH = 1.0 / 3.6e6

CO2_P1, CO2_P2, CO2_P3 = 2.0, 10.0, 50.0
TEMP_P1, TEMP_P2, TEMP_P3 = 1.0, 5.0, 25.0

COLUMN_NAMES = [
    "Time", "Outdoor_Tdb_C", "Outdoor_Twb_C",
    "Space1_occupants", "Space2_occupants", "Space3_occupants",
    "Space4_occupants", "Space5_occupants",
    "lights-1", "lights-2", "lights-3", "lights-4", "lights-5",
    "equip-1", "equip-2", "equip-3", "equip-4", "equip-5",
    "Plenum1_T_C", "Plenum1_RH_%",
    "Space1_T_C", "Space1_RH_%", "Space2_T_C", "Space2_RH_%",
    "Space3_T_C", "Space3_RH_%", "Space4_T_C", "Space4_RH_%",
    "Space5_T_C", "Space5_RH_%",
    "Plenum_CO2_ppm", "Plenum_CO2_pred", "Plenum_CO2_setpoint_ppm", "Plenum_CO2_internal_gain",
    "Space1_CO2_ppm", "Space1_CO2_pred", "Space1_CO2_setpoint_ppm", "Space1_CO2_internal_gain",
    "Space2_CO2_ppm", "Space2_CO2_pred", "Space2_CO2_setpoint_ppm", "Space2_CO2_internal_gain",
    "Space3_CO2_ppm", "Space3_CO2_pred", "Space3_CO2_setpoint_ppm", "Space3_CO2_internal_gain",
    "Space4_CO2_ppm", "Space4_CO2_pred", "Space4_CO2_setpoint_ppm", "Space4_CO2_internal_gain",
    "Space5_CO2_ppm", "Space5_CO2_pred", "Space5_CO2_setpoint_ppm", "Space5_CO2_internal_gain",
    "doas_fan", "fcu_1", "fcu_2", "fcu_3", "fcu_4", "fcu_5",
    "hex", "chiller", "tower", "boiler", "coldw_pump", "condw_pump",
    "hotw_pump",
    "Node2_T_C", "Node2_Mdot_kg/s", "Node2_W_Ratio",
    "Node2_SP_T_C", "Node2_CO2_ppm", "Node1_T_C",
    "Gas_Facility_E_J", "Elec_Facility_E_J", "Elec_HVAC_E_J",
    "CoolingCoils:EnergyTransfer", "HeatingCoils:EnergyTransfer",
    "ElectricityNet:Facility", "General:Cooling:EnergyTransfer", "Cooling:EnergyTransfer",
]

ZONE_CO2_COLS = [f"Space{i}_CO2_ppm" for i in range(1, 6)]
ZONE_TEMP_COLS = [f"Space{i}_T_C" for i in range(1, 6)]
BEST_COST = float("inf")


def _safe_float(v: float) -> float:
    return float(np.round(v, 6))


def load_eplusout(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date/Time"] = df["Date/Time"].astype(str).str.strip()
    mask_24 = df["Date/Time"].str.contains("24:00:00")
    df.loc[mask_24, "Date/Time"] = df.loc[mask_24, "Date/Time"].str.replace("24:00:00", "00:00:00")
    df["Date/Time"] = "2024/" + df["Date/Time"]
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%Y/%m/%d %H:%M:%S")
    df.loc[mask_24, "Date/Time"] = df.loc[mask_24, "Date/Time"] + pd.Timedelta(days=1)

    # Keep Date/Time in column 0 and map known output columns safely.
    current_cols = list(df.columns)
    if "Date/Time" not in current_cols and len(current_cols) > 0:
        current_cols[0] = "Date/Time"
        df.columns = current_cols
    if len(df.columns) >= 2:
        metric_count = min(len(df.columns) - 1, len(COLUMN_NAMES) - 1)
        renamed = ["Date/Time"] + COLUMN_NAMES[1:1 + metric_count]
        if len(df.columns) > 1 + metric_count:
            renamed += list(df.columns[1 + metric_count :])
        df.columns = renamed
    return df


def _timestep_hours(df: pd.DataFrame) -> float:
    if "Date/Time" not in df.columns:
        return 0.25
    diffs = df["Date/Time"].diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.25
    return float(np.median(diffs) / 3600.0)


def _steps_per_day(step_hours: float) -> int:
    return max(1, int(round(24.0 / max(step_hours, 1e-9))))


def compute_temperature_limits(df: pd.DataFrame, steps_per_day: int) -> Tuple[np.ndarray, ...]:
    t_mean = pd.to_numeric(df["Outdoor_Tdb_C"], errors="coerce").rolling(
        steps_per_day, min_periods=steps_per_day
    ).mean()
    t = t_mean.to_numpy(dtype=float)

    lower_s1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_s1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.0 + 0.2 * t, 25.0))
    lower_s2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_s2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20 * t, 26.0))
    lower_s3 = np.full_like(t, 20.0)
    upper_s3 = np.where(t <= 10, 25.0, 27.0)
    return lower_s1, upper_s1, lower_s2, upper_s2, lower_s3, upper_s3


def compute_total_cost(df: pd.DataFrame) -> Dict[str, float]:
    step_hours = _timestep_hours(df)
    steps_day = _steps_per_day(step_hours)

    elec_kwh = pd.to_numeric(df["Elec_Facility_E_J"], errors="coerce").fillna(0.0).sum() * J_TO_KWH
    gas_kwh = pd.to_numeric(df["Gas_Facility_E_J"], errors="coerce").fillna(0.0).sum() * J_TO_KWH
    energy = elec_kwh * ELEC_PRICE + gas_kwh * GAS_PRICE

    co2_penalty = 0.0
    for col in ZONE_CO2_COLS:
        co2 = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        co2_penalty += CO2_P1 * step_hours * ((co2 > 770) & (co2 <= 970)).sum()
        co2_penalty += CO2_P2 * step_hours * ((co2 > 970) & (co2 <= 1220)).sum()
        co2_penalty += CO2_P3 * step_hours * (co2 > 1220).sum()

    lower_s1, upper_s1, lower_s2, upper_s2, lower_s3, upper_s3 = compute_temperature_limits(df, steps_day)
    temp_penalty = 0.0
    for col in ZONE_TEMP_COLS:
        t_in = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        in_s1 = (t_in >= lower_s1) & (t_in <= upper_s1)
        in_s2 = (t_in >= lower_s2) & (t_in <= upper_s2)
        in_s3 = (t_in >= lower_s3) & (t_in <= upper_s3)
        temp_penalty += TEMP_P1 * step_hours * (~in_s1 & in_s2).sum()
        temp_penalty += TEMP_P2 * step_hours * (~in_s2 & in_s3).sum()
        temp_penalty += TEMP_P3 * step_hours * (~in_s3).sum()

    total = energy + co2_penalty + temp_penalty
    return {
        "energy_cost": float(energy),
        "co2_penalty": float(co2_penalty),
        "temp_penalty": float(temp_penalty),
        "step_hours": float(step_hours),
        "total_cost": float(total),
    }


def write_best_files(params: Dict[str, float], costs: Dict[str, float], trial_num: int, elapsed_s: float) -> None:
    serializable = {k: _safe_float(v) for k, v in params.items()}
    BEST_JSON.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")

    lines = [
        '"""Current best parameters for vesa_6 (auto-updated by optimize.py)."""',
        "",
        "BEST_PARAMS = {",
    ]
    for key in sorted(serializable.keys()):
        lines.append(f'    "{key}": {serializable[key]},')
    lines.append("}")
    lines.append("")
    BEST_PY.write_text("\n".join(lines), encoding="utf-8")

    report = [
        f"trial={trial_num}",
        f"elapsed_s={elapsed_s:.2f}",
        f"total_cost={costs['total_cost']:.4f}",
        f"energy_cost={costs['energy_cost']:.4f}",
        f"co2_penalty={costs['co2_penalty']:.4f}",
        f"temp_penalty={costs['temp_penalty']:.4f}",
        f"step_hours={costs['step_hours']:.6f}",
        "",
        "parameters:",
    ]
    report.extend([f"{k}={serializable[k]}" for k in sorted(serializable.keys())])
    BEST_TXT.write_text("\n".join(report) + "\n", encoding="utf-8")


def sample_params(trial: optuna.Trial) -> Dict[str, float]:
    # Fast-converging reduced search space:
    # keep less-sensitive schedule/geometry parameters fixed and optimize only
    # key energy/comfort/IAQ trade-off knobs.
    params = {
        # Fixed defaults
        "preflush_start": 5.5,
        "work_start": 7.0,
        "work_end": 18.0,
        "evening_end": 21.0,
        "return_air_cold_ref": 21.0,
        "return_air_warm_ref": 25.0,
        "flow_co2_boost": 1.0,
        "cold_limit_low": -25.0,
        "cold_limit_high": -12.0,
        "nightflush_delta": 1.5,
        "nightflush_min_temp": 22.0,
        "min_deadband": 1.0,
        # Tuned dimensions
        "htg_margin": trial.suggest_float("htg_margin", 0.22, 0.60),
        "clg_margin": trial.suggest_float("clg_margin", 0.18, 0.55),
        "night_htg_setback": trial.suggest_float("night_htg_setback", 0.05, 0.45),
        "night_clg_setup": trial.suggest_float("night_clg_setup", 0.05, 0.45),
        "sup_temp_cold": trial.suggest_float("sup_temp_cold", 19.0, 21.0),
        "sup_temp_warm": trial.suggest_float("sup_temp_warm", 16.0, 18.2),
        "flow_work": trial.suggest_float("flow_work", 0.18, 0.34),
        "flow_evening": trial.suggest_float("flow_evening", 0.06, 0.20),
        "flow_night": trial.suggest_float("flow_night", 0.0, 0.07),
        "flow_preflush": trial.suggest_float("flow_preflush", 0.20, 0.48),
        "co2_low": trial.suggest_float("co2_low", 560.0, 690.0),
        "co2_high": trial.suggest_float("co2_high", 720.0, 860.0),
        "co2_emergency": trial.suggest_float("co2_emergency", 860.0, 1040.0),
        "nightflush_flow": trial.suggest_float("nightflush_flow", 0.25, 0.65),
        "supply_override_gap": trial.suggest_float("supply_override_gap", 0.25, 1.0),
        "temp_recovery_gap": trial.suggest_float("temp_recovery_gap", 0.25, 0.65),
        "temp_recovery_flow": trial.suggest_float("temp_recovery_flow", 0.45, 0.85),
        "s2_recovery_flow": trial.suggest_float("s2_recovery_flow", 0.70, 1.0),
        "s2_recovery_margin": trial.suggest_float("s2_recovery_margin", 0.0, 0.8),
    }

    # Repair constraints for physically meaningful and stable settings.
    if params["clg_margin"] < params["htg_margin"] - 0.1:
        params["clg_margin"] = params["htg_margin"] - 0.1
    params["clg_margin"] = max(0.1, min(0.9, params["clg_margin"]))

    if params["work_end"] <= params["work_start"] + 6.0:
        params["work_end"] = params["work_start"] + 6.0
    if params["evening_end"] < params["work_end"]:
        params["evening_end"] = params["work_end"]
    if params["preflush_start"] > params["work_start"] - 0.25:
        params["preflush_start"] = params["work_start"] - 0.25

    if params["sup_temp_cold"] <= params["sup_temp_warm"] + 0.2:
        params["sup_temp_cold"] = params["sup_temp_warm"] + 0.2
    if params["return_air_warm_ref"] <= params["return_air_cold_ref"] + 0.5:
        params["return_air_warm_ref"] = params["return_air_cold_ref"] + 0.5

    if params["flow_evening"] > params["flow_work"]:
        params["flow_evening"] = params["flow_work"]
    if params["flow_night"] > params["flow_evening"]:
        params["flow_night"] = params["flow_evening"]
    if params["flow_preflush"] < params["flow_work"]:
        params["flow_preflush"] = params["flow_work"]
    if params["flow_co2_boost"] < params["flow_work"] + 0.1:
        params["flow_co2_boost"] = min(1.0, params["flow_work"] + 0.1)

    if params["co2_high"] <= params["co2_low"] + 40.0:
        params["co2_high"] = params["co2_low"] + 40.0
    if params["co2_emergency"] <= params["co2_high"] + 20.0:
        params["co2_emergency"] = params["co2_high"] + 20.0

    if params["cold_limit_high"] <= params["cold_limit_low"] + 2.0:
        params["cold_limit_high"] = params["cold_limit_low"] + 2.0

    if params["temp_recovery_flow"] < params["flow_work"]:
        params["temp_recovery_flow"] = params["flow_work"]
    if params["s2_recovery_flow"] < params["temp_recovery_flow"]:
        params["s2_recovery_flow"] = params["temp_recovery_flow"]
    if params["s2_recovery_margin"] < 0.0:
        params["s2_recovery_margin"] = 0.0

    return params


def objective(trial: optuna.Trial) -> float:
    global BEST_COST

    params = sample_params(trial)

    from pyenergyplus.api import EnergyPlusAPI
    from energyplus_controller import EnergyPlusController
    from my_model import Vesa6Model

    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    model = Vesa6Model(**params)
    controller = EnergyPlusController(api, model)

    api.runtime.callback_after_new_environment_warmup_complete(state, controller.initialize_handles)
    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, controller.control_callback)

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
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
        raise optuna.TrialPruned()

    df = load_eplusout(csv_path)
    if len(df) < 8000:
        # Guard against incomplete yearly runs caused by inconsistent runtime context.
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
        raise optuna.TrialPruned()
    costs = compute_total_cost(df)
    trial.set_user_attr("energy_cost", costs["energy_cost"])
    trial.set_user_attr("co2_penalty", costs["co2_penalty"])
    trial.set_user_attr("temp_penalty", costs["temp_penalty"])
    trial.set_user_attr("step_hours", costs["step_hours"])
    trial.set_user_attr("elapsed_s", elapsed)

    print(
        f"Trial {trial.number:04d} -> total={costs['total_cost']:.2f}€ "
        f"(E={costs['energy_cost']:.2f}, CO2={costs['co2_penalty']:.2f}, T={costs['temp_penalty']:.2f}) "
        f"[{elapsed:.1f}s]"
    )

    if costs["total_cost"] < BEST_COST:
        BEST_COST = costs["total_cost"]
        write_best_files(params, costs, trial.number, elapsed)

        if BEST_RUN_DIR.exists():
            shutil.rmtree(BEST_RUN_DIR, ignore_errors=True)
        shutil.copytree(trial_dir, BEST_RUN_DIR)

        print("=" * 76)
        print(f"NEW BEST FOUND at trial {trial.number}: {BEST_COST:.2f} €")
        print(f"Saved: {BEST_JSON.name}, {BEST_PY.name}, {BEST_TXT.name}, and {BEST_RUN_DIR}")
        print("=" * 76)

    if trial_dir.exists():
        shutil.rmtree(trial_dir, ignore_errors=True)

    return costs["total_cost"]


def main() -> None:
    global BEST_COST

    # Ensure consistent runtime context regardless of where the script is launched from.
    os.chdir(SCRIPT_DIR)
    OPTUNA_OUT.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        constant_liar=True,
        n_startup_trials=8,
    )
    study = optuna.create_study(
        study_name="vesa6_hvac_optimization_full_year",
        storage=f"sqlite:///{OPTUNA_OUT}/optuna_study.db",
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    if len(study.trials) > 0:
        try:
            BEST_COST = float(study.best_value)
            best_attrs = study.best_trial.user_attrs
            best_costs = {
                "total_cost": float(study.best_value),
                "energy_cost": float(best_attrs.get("energy_cost", float("nan"))),
                "co2_penalty": float(best_attrs.get("co2_penalty", float("nan"))),
                "temp_penalty": float(best_attrs.get("temp_penalty", float("nan"))),
                "step_hours": float(best_attrs.get("step_hours", 0.25)),
            }
            write_best_files(
                study.best_params,
                best_costs,
                study.best_trial.number,
                float(best_attrs.get("elapsed_s", 0.0)),
            )
        except Exception:
            BEST_COST = float("inf")

    # Seed the search with strong manual baselines so TPE starts near useful regions.
    seed_current = {
        "htg_margin": 0.35, "clg_margin": 0.30, "night_htg_setback": 0.25, "night_clg_setup": 0.25,
        "sup_temp_cold": 20.5, "sup_temp_warm": 17.0, "flow_work": 0.22, "flow_evening": 0.10,
        "flow_night": 0.02, "flow_preflush": 0.28, "co2_low": 610.0, "co2_high": 770.0,
        "co2_emergency": 930.0, "nightflush_flow": 0.35, "temp_recovery_gap": 0.40,
        "temp_recovery_flow": 0.55, "s2_recovery_flow": 0.85, "supply_override_gap": 0.50,
        "s2_recovery_margin": 0.0,
    }
    seed_energy = {
        "htg_margin": 0.28, "clg_margin": 0.24, "night_htg_setback": 0.35, "night_clg_setup": 0.35,
        "sup_temp_cold": 20.0, "sup_temp_warm": 17.2, "flow_work": 0.20, "flow_evening": 0.08,
        "flow_night": 0.01, "flow_preflush": 0.24, "co2_low": 630.0, "co2_high": 800.0,
        "co2_emergency": 980.0, "nightflush_flow": 0.30, "temp_recovery_gap": 0.50,
        "temp_recovery_flow": 0.50, "s2_recovery_flow": 0.78, "supply_override_gap": 0.75,
        "s2_recovery_margin": 0.45,
    }
    seed_comfort = {
        "htg_margin": 0.48, "clg_margin": 0.42, "night_htg_setback": 0.12, "night_clg_setup": 0.12,
        "sup_temp_cold": 20.8, "sup_temp_warm": 16.6, "flow_work": 0.28, "flow_evening": 0.16,
        "flow_night": 0.04, "flow_preflush": 0.36, "co2_low": 590.0, "co2_high": 750.0,
        "co2_emergency": 900.0, "nightflush_flow": 0.45, "temp_recovery_gap": 0.33,
        "temp_recovery_flow": 0.66, "s2_recovery_flow": 0.92, "supply_override_gap": 0.40,
        "s2_recovery_margin": 0.05,
    }
    seed_energy_keep_comfort_1 = {
        "htg_margin": 0.42, "clg_margin": 0.38, "night_htg_setback": 0.20, "night_clg_setup": 0.16,
        "sup_temp_cold": 19.8, "sup_temp_warm": 16.4, "flow_work": 0.24, "flow_evening": 0.13,
        "flow_night": 0.03, "flow_preflush": 0.31, "co2_low": 600.0, "co2_high": 760.0,
        "co2_emergency": 910.0, "nightflush_flow": 0.38, "temp_recovery_gap": 0.36,
        "temp_recovery_flow": 0.60, "s2_recovery_flow": 0.88, "supply_override_gap": 0.60,
        "s2_recovery_margin": 0.20,
    }
    seed_energy_keep_comfort_2 = {
        "htg_margin": 0.39, "clg_margin": 0.35, "night_htg_setback": 0.24, "night_clg_setup": 0.20,
        "sup_temp_cold": 19.5, "sup_temp_warm": 16.3, "flow_work": 0.23, "flow_evening": 0.12,
        "flow_night": 0.02, "flow_preflush": 0.30, "co2_low": 605.0, "co2_high": 770.0,
        "co2_emergency": 930.0, "nightflush_flow": 0.36, "temp_recovery_gap": 0.40,
        "temp_recovery_flow": 0.57, "s2_recovery_flow": 0.86, "supply_override_gap": 0.68,
        "s2_recovery_margin": 0.30,
    }
    seed_balance = {
        "htg_margin": 0.44, "clg_margin": 0.39, "night_htg_setback": 0.16, "night_clg_setup": 0.16,
        "sup_temp_cold": 20.0, "sup_temp_warm": 16.5, "flow_work": 0.25, "flow_evening": 0.14,
        "flow_night": 0.03, "flow_preflush": 0.33, "co2_low": 595.0, "co2_high": 755.0,
        "co2_emergency": 905.0, "nightflush_flow": 0.40, "temp_recovery_gap": 0.34,
        "temp_recovery_flow": 0.62, "s2_recovery_flow": 0.90, "supply_override_gap": 0.50,
        "s2_recovery_margin": 0.10,
    }
    study.enqueue_trial(seed_current)
    study.enqueue_trial(seed_energy)
    study.enqueue_trial(seed_comfort)
    study.enqueue_trial(seed_energy_keep_comfort_1)
    study.enqueue_trial(seed_energy_keep_comfort_2)
    study.enqueue_trial(seed_balance)

    n_trials = int(os.environ.get("VESA6_TRIALS", "80"))
    print(f"Starting vesa_6 optimization ({n_trials} trials)")
    print(f"Current best in study: {BEST_COST if np.isfinite(BEST_COST) else 'none'}")
    print(f"Target total cost: < {TARGET_TOTAL:.1f}")
    print("Best parameter files are updated automatically when a new best is found.")

    try:
        def _stop_on_target(study_obj: optuna.Study, _trial: optuna.Trial) -> None:
            if study_obj.best_value <= TARGET_TOTAL:
                print(f"Target reached: best={study_obj.best_value:.2f} <= {TARGET_TOTAL:.2f}. Stopping.")
                study_obj.stop()

        study.optimize(objective, n_trials=n_trials, callbacks=[_stop_on_target])
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\n" + "=" * 76)
    print("Optimization finished.")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best total cost: {study.best_value:.2f} €")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.6f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
