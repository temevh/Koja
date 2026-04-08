"""Hackathon scoring function — computes total annual cost from eplusout.csv.

Copied from nibs_bo_002/scoring.py for standalone use within nibs_mpc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Column mapping (must match the IDF output order) ────────────────────────

COLUMN_NAMES = [
    "Time",
    "Outdoor_Tdb_C",
    "Outdoor_Twb_C",
    "Space1_occupants", "Space2_occupants", "Space3_occupants",
    "Space4_occupants", "Space5_occupants",
    "lights-1", "lights-2", "lights-3", "lights-4", "lights-5",
    "equip-1", "equip-2", "equip-3", "equip-4", "equip-5",
    "Plenum1_T_C", "Plenum1_RH_%",
    "Space1_T_C", "Space1_RH_%",
    "Space2_T_C", "Space2_RH_%",
    "Space3_T_C", "Space3_RH_%",
    "Space4_T_C", "Space4_RH_%",
    "Space5_T_C", "Space5_RH_%",
    "Plenum_CO2_ppm", "Plenum_CO2_pred", "Plenum_CO2_setpoint_ppm", "Plenum_CO2_internal_gain",
    "Space1_CO2_ppm", "Space1_CO2_pred", "Space1_CO2_setpoint_ppm", "Space1_CO2_internal_gain",
    "Space2_CO2_ppm", "Space2_CO2_pred", "Space2_CO2_setpoint_ppm", "Space2_CO2_internal_gain",
    "Space3_CO2_ppm", "Space3_CO2_pred", "Space3_CO2_setpoint_ppm", "Space3_CO2_internal_gain",
    "Space4_CO2_ppm", "Space4_CO2_pred", "Space4_CO2_setpoint_ppm", "Space4_CO2_internal_gain",
    "Space5_CO2_ppm", "Space5_CO2_pred", "Space5_CO2_setpoint_ppm", "Space5_CO2_internal_gain",
    "doas_fan",
    "fcu_1", "fcu_2", "fcu_3", "fcu_4", "fcu_5",
    "hex", "chiller", "tower", "boiler",
    "coldw_pump", "condw_pump", "hotw_pump",
    "Node2_T_C", "Node2_Mdot_kg/s", "Node2_W_Ratio",
    "Node2_SP_T_C", "Node2_CO2_ppm", "Node1_T_C",
    "Gas_Facility_E_J", "Elec_Facility_E_J", "Elec_HVAC_E_J",
    "CoolingCoils:EnergyTransfer", "HeatingCoils:EnergyTransfer",
    "ElectricityNet:Facility",
    "General:Cooling:EnergyTransfer", "Cooling:EnergyTransfer",
]

ELEC_PRICE = 0.11   # €/kWh
GAS_PRICE = 0.06    # €/kWh
J_TO_KWH = 1 / 3.6e6

ZONE_CO2_COLS = [f"Space{i}_CO2_ppm" for i in range(1, 6)]

CO2_P1 = 2    # above S1 (770) but ≤ S2 (970)
CO2_P2 = 10   # above S2 (970) but ≤ S3 (1220)
CO2_P3 = 50   # above S3 (1220)

TEMP_P1 = 1    # outside S1 but inside S2
TEMP_P2 = 5    # outside S2 but inside S3
TEMP_P3 = 25   # outside S3


def load_eplusout(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date/Time"] = df["Date/Time"].str.strip()

    mask_24 = df["Date/Time"].str.contains("24:00:00")
    df.loc[mask_24, "Date/Time"] = df.loc[mask_24, "Date/Time"].str.replace(
        "24:00:00", "00:00:00"
    )
    df["Date/Time"] = "2024/" + df["Date/Time"]
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%Y/%m/%d %H:%M:%S")
    df.loc[mask_24, "Date/Time"] += pd.Timedelta(days=1)

    df.columns = COLUMN_NAMES
    return df


def _energy_cost(df: pd.DataFrame) -> float:
    elec_kwh = df["Elec_Facility_E_J"].sum() * J_TO_KWH
    gas_kwh = df["Gas_Facility_E_J"].sum() * J_TO_KWH
    return elec_kwh * ELEC_PRICE + gas_kwh * GAS_PRICE


def _co2_penalty(df: pd.DataFrame) -> float:
    total = 0.0
    for col in ZONE_CO2_COLS:
        co2 = df[col]
        h_s1_s2 = ((co2 > 770) & (co2 <= 970)).sum()
        h_s2_s3 = ((co2 > 970) & (co2 <= 1220)).sum()
        h_above = (co2 > 1220).sum()
        total += CO2_P1 * h_s1_s2 + CO2_P2 * h_s2_s3 + CO2_P3 * h_above
    return total


def _temperature_penalty(df: pd.DataFrame) -> float:
    t_mean = df["Outdoor_Tdb_C"].rolling(24, min_periods=24).mean()
    t = pd.to_numeric(t_mean, errors="coerce").to_numpy(dtype=float)

    lower_S1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_S1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.5 + 0.166 * t, 25.0))
    lower_S2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_S2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20 * t, 26.0))
    lower_S3 = np.full_like(t, 20.0)
    upper_S3 = np.where(t <= 10, 25.0, 27.0)

    total = 0.0
    for i in range(1, 6):
        col = f"Space{i}_T_C"
        t_in = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        in_s1 = (t_in >= lower_S1) & (t_in <= upper_S1)
        in_s2 = (t_in >= lower_S2) & (t_in <= upper_S2)
        in_s3 = (t_in >= lower_S3) & (t_in <= upper_S3)

        h_s1_only = (~in_s1 & in_s2).sum()
        h_s2_only = (~in_s2 & in_s3).sum()
        h_out = (~in_s3).sum()

        total += TEMP_P1 * h_s1_only + TEMP_P2 * h_s2_only + TEMP_P3 * h_out
    return total


def compute_total_cost(csv_path: str) -> dict:
    """Load eplusout.csv and return the full cost breakdown."""
    df = load_eplusout(csv_path)
    energy = _energy_cost(df)
    co2 = _co2_penalty(df)
    temp = _temperature_penalty(df)
    return {
        "energy_cost_eur": energy,
        "co2_penalty_eur": co2,
        "temp_penalty_eur": temp,
        "total_cost_eur": energy + co2 + temp,
    }
