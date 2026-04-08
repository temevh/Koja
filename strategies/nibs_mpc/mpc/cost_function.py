"""Per-timestep cost function for MPC planning.

Implements the exact competition scoring formula adapted for single
15-minute timestep evaluation with per-zone state inputs. Used during
CEM rollouts to evaluate candidate action sequences.
"""

from __future__ import annotations

import numpy as np


# Prices
ELEC_PRICE = 0.11   # €/kWh
GAS_PRICE = 0.06    # €/kWh
J_TO_KWH = 1 / 3.6e6

# CO2 penalty per zone per hour (we multiply by 0.25 for 15-min step)
CO2_THRESHOLDS = np.array([770.0, 970.0, 1220.0])
CO2_PENALTIES = np.array([2.0, 10.0, 50.0])   # €/h per zone

# Temperature penalty per zone per hour (also × 0.25)
TEMP_P1 = 1.0    # outside S1 but inside S2
TEMP_P2 = 5.0    # outside S2 but inside S3
TEMP_P3 = 25.0   # outside S3

TIMESTEP_FRACTION = 0.25  # 15 min / 60 min


def compute_comfort_bands(t_out_24h: float | np.ndarray):
    """Compute S1/S2/S3 comfort bands from 24h rolling outdoor temp.

    Returns: (lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3)
    """
    t = np.asarray(t_out_24h, dtype=np.float64)

    lower_S1 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.075 * t, 22.0))
    upper_S1 = np.where(t <= 0, 22.0, np.where(t <= 15, 22.5 + 0.166 * t, 25.0))
    lower_S2 = np.where(t <= 0, 20.5, np.where(t <= 20, 20.5 + 0.025 * t, 21.0))
    upper_S2 = np.where(t <= 0, 23.0, np.where(t <= 15, 23.0 + 0.20 * t, 26.0))
    lower_S3 = np.full_like(t, 20.0)
    upper_S3 = np.where(t <= 10, 25.0, 27.0)

    return lower_S1, upper_S1, lower_S2, upper_S2, lower_S3, upper_S3


def compute_step_cost(
    zone_temps: np.ndarray,
    zone_co2: np.ndarray,
    elec_j: float | np.ndarray,
    gas_j: float | np.ndarray,
    t_out_24h: float,
) -> float | np.ndarray:
    """Compute cost (€) for a single 15-minute timestep.

    Args:
        zone_temps: shape (5,) or (batch, 5) — zone temperatures [°C]
        zone_co2: shape (5,) or (batch, 5) — zone CO2 [ppm]
        elec_j: per-step electricity [J]
        gas_j: per-step gas [J]
        t_out_24h: 24h rolling mean outdoor temp [°C]

    Returns:
        Total cost for this timestep [€]. Scalar or (batch,).
    """
    zone_temps = np.asarray(zone_temps, dtype=np.float64)
    zone_co2 = np.asarray(zone_co2, dtype=np.float64)
    elec_j = np.asarray(elec_j, dtype=np.float64)
    gas_j = np.asarray(gas_j, dtype=np.float64)

    # Energy cost
    energy_cost = elec_j * J_TO_KWH * ELEC_PRICE + gas_j * J_TO_KWH * GAS_PRICE

    # CO2 penalty (per zone, scaled to 15-min)
    # For each zone: find which tier the CO2 falls in
    co2_penalty = np.zeros_like(elec_j)
    for z in range(5):
        if zone_co2.ndim == 1:
            co2_val = zone_co2[z]
        else:
            co2_val = zone_co2[..., z]

        p = np.where(co2_val > 1220, 50.0,
                np.where(co2_val > 970, 10.0,
                    np.where(co2_val > 770, 2.0, 0.0)))
        co2_penalty = co2_penalty + p * TIMESTEP_FRACTION

    # Temperature penalty (per zone, scaled to 15-min)
    l1, u1, l2, u2, l3, u3 = compute_comfort_bands(t_out_24h)

    temp_penalty = np.zeros_like(elec_j)
    for z in range(5):
        if zone_temps.ndim == 1:
            t_in = zone_temps[z]
        else:
            t_in = zone_temps[..., z]

        in_s1 = (t_in >= l1) & (t_in <= u1)
        in_s2 = (t_in >= l2) & (t_in <= u2)
        in_s3 = (t_in >= l3) & (t_in <= u3)

        p = np.where(~in_s3, TEMP_P3,
                np.where(~in_s2, TEMP_P2,
                    np.where(~in_s1, TEMP_P1, 0.0)))
        temp_penalty = temp_penalty + p * TIMESTEP_FRACTION

    return energy_cost + co2_penalty + temp_penalty


def compute_step_cost_batch(
    zone_temps: np.ndarray,
    zone_co2: np.ndarray,
    elec_j: np.ndarray,
    gas_j: np.ndarray,
    t_out_24h: float,
) -> np.ndarray:
    """Vectorized cost for a batch of samples.

    Args:
        zone_temps: shape (batch, 5)
        zone_co2: shape (batch, 5)
        elec_j: shape (batch,)
        gas_j: shape (batch,)
        t_out_24h: scalar

    Returns:
        shape (batch,) — cost per sample
    """
    return compute_step_cost(zone_temps, zone_co2, elec_j, gas_j, t_out_24h)
