#!/usr/bin/env python3
"""Score a single EnergyPlus run: energy cost + CO2 penalty + temperature penalty."""

import sys
import pandas as pd
import numpy as np

# ── Column mapping ──────────────────────────────────────────────────────────

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

ELEC_PRICE = 0.11   # €/kWh
GAS_PRICE  = 0.06   # €/kWh
J_TO_KWH   = 1 / 3.6e6

ZONE_CO2_COLS   = [f'Space{i}_CO2_ppm' for i in range(1, 6)]
CLASS_THRESHOLD = 0.90

CO2_P1 = 2    # €/h — above 770 ppm but ≤ 970 ppm
CO2_P2 = 10   # €/h — above 970 ppm but ≤ 1220 ppm
CO2_P3 = 50   # €/h — above 1220 ppm

TEMP_P1 = 1   # €/h — outside S1 but inside S2
TEMP_P2 = 5   # €/h — outside S2 but inside S3
TEMP_P3 = 25  # €/h — outside S3


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


def compute_energy_cost(df: pd.DataFrame) -> dict:
    df = df.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    monthly = df.groupby(df['Time'].dt.to_period('M')).agg(
        elec_fac_kWh=('Elec_Facility_E_J', lambda x: x.sum() * J_TO_KWH),
        gas_kWh=('Gas_Facility_E_J', lambda x: x.sum() * J_TO_KWH),
    )
    elec_cost = (monthly['elec_fac_kWh'] * ELEC_PRICE).sum()
    gas_cost = (monthly['gas_kWh'] * GAS_PRICE).sum()
    return {
        'elec_kWh': monthly['elec_fac_kWh'].sum(),
        'gas_kWh': monthly['gas_kWh'].sum(),
        'elec_cost_eur': elec_cost,
        'gas_cost_eur': gas_cost,
        'energy_cost_eur': elec_cost + gas_cost,
    }


def compute_co2_penalty(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ZONE_CO2_COLS:
        co2 = df[col]
        h_s1_s2 = ((co2 > 770) & (co2 <= 970)).sum()
        h_s2_s3 = ((co2 > 970) & (co2 <= 1220)).sum()
        h_above = (co2 > 1220).sum()
        cost = CO2_P1 * h_s1_s2 + CO2_P2 * h_s2_s3 + CO2_P3 * h_above
        rows.append({
            'space': col.replace('_CO2_ppm', ''),
            'h_above_S1': h_s1_s2,
            'h_above_S2': h_s2_s3,
            'h_above_S3': h_above,
            'penalty_eur': cost,
        })
    return pd.DataFrame(rows).set_index('space')


def compute_temperature_limits(df):
    t_mean = df['Outdoor_Tdb_C'].rolling(24, min_periods=24).mean()
    t = pd.to_numeric(t_mean, errors='coerce').to_numpy(dtype=float)
    lower_S1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_S1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.5 + 0.166 * t, 25.0))
    lower_S2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_S2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20 * t, 26.0))
    lower_S3 = np.full_like(t, 20.0)
    upper_S3 = np.where(t <= 10, 25.0, 27.0)
    return lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3


def compute_temperature_penalty(df: pd.DataFrame) -> pd.DataFrame:
    lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3 = compute_temperature_limits(df)
    rows = []
    for i in range(1, 6):
        t_in = pd.to_numeric(df[f'Space{i}_T_C'], errors='coerce').to_numpy(dtype=float)
        total = len(t_in)
        in_s1 = (t_in >= lower_S1) & (t_in <= upper_S1)
        in_s2 = (t_in >= lower_S2) & (t_in <= upper_S2)
        in_s3 = (t_in >= lower_S3) & (t_in <= upper_S3)
        h_s1_only = (~in_s1 & in_s2).sum()
        h_s2_only = (~in_s2 & in_s3).sum()
        h_out = (~in_s3).sum()
        penalty = TEMP_P1 * h_s1_only + TEMP_P2 * h_s2_only + TEMP_P3 * h_out
        pct_s1 = 100 * in_s1.sum() / total if total else 0
        pct_s2 = 100 * in_s2.sum() / total if total else 0
        pct_s3 = 100 * in_s3.sum() / total if total else 0
        pct_req = CLASS_THRESHOLD * 100
        if pct_s1 >= pct_req:
            achieved = 'S1'
        elif pct_s2 >= pct_req:
            achieved = 'S2'
        elif pct_s3 >= pct_req:
            achieved = 'S3'
        else:
            achieved = 'Out of class'
        rows.append({
            'space': f'Space{i}',
            'achieved_class': achieved,
            'h_outside_S1': h_s1_only,
            'h_outside_S2': h_s2_only,
            'h_outside_S3': h_out,
            'S1_%': pct_s1,
            'penalty_eur': penalty,
        })
    return pd.DataFrame(rows).set_index('space')


def score(csv_path: str):
    df = load_eplusout(csv_path)

    energy = compute_energy_cost(df)
    co2_pen = compute_co2_penalty(df)
    temp_pen = compute_temperature_penalty(df)

    co2_total = co2_pen['penalty_eur'].sum()
    temp_total = temp_pen['penalty_eur'].sum()
    total = energy['energy_cost_eur'] + co2_total + temp_total

    print(f'=== Score: {csv_path} ===\n')

    print(f'Energy:  elec {energy["elec_kWh"]:.1f} kWh, gas {energy["gas_kWh"]:.1f} kWh')
    print(f'         elec {energy["elec_cost_eur"]:.2f} €, gas {energy["gas_cost_eur"]:.2f} €')
    print(f'         energy cost = {energy["energy_cost_eur"]:.2f} €\n')

    print('CO2 penalty per space:')
    print(co2_pen.to_string())
    print(f'CO2 penalty total = {co2_total:.2f} €\n')

    print('Temperature penalty per space:')
    print(temp_pen.to_string())
    print(f'Temperature penalty total = {temp_total:.2f} €\n')

    print(f'─────────────────────────────')
    print(f'Energy cost:       {energy["energy_cost_eur"]:>10.2f} €')
    print(f'CO2 penalty:       {co2_total:>10.2f} €')
    print(f'Temperature penalty:{temp_total:>9.2f} €')
    print(f'TOTAL COST:        {total:>10.2f} €')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <path/to/eplusout.csv>')
        sys.exit(1)
    score(sys.argv[1])
