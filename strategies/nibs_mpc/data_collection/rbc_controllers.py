"""RBC controllers for MPC training data collection.

Two controller types:
  - BOParamRBC: Uses the top 20 BO-optimized parameter sets from nibs_bo_002
  - FullOnRBC: rbc_full_on style with randomized setpoints/flows for exploration
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


RING_BUFFER_SIZE = 96  # 24h at 15-min timesteps

# Action bounds
HTG_MIN, HTG_MAX = 18.0, 25.0
CLG_MIN, CLG_MAX = 18.0, 25.0
SUPPLY_MIN, SUPPLY_MAX = 16.0, 21.0
FLOW_MIN, FLOW_MAX = 0.0, 1.0


def _s1_band(t_out_24h: float) -> Tuple[float, float]:
    """Compute S1 lower/upper from 24h rolling outdoor temp."""
    if t_out_24h <= 0:
        lower, upper = 20.5, 22.0
    elif t_out_24h <= 15:
        lower = 20.5 + 0.075 * t_out_24h
        upper = 22.5 + 0.166 * t_out_24h
    elif t_out_24h <= 20:
        lower = 20.5 + 0.075 * t_out_24h
        upper = 25.0
    else:
        lower, upper = 22.0, 25.0
    return lower, upper


def _clip_actions(
    htg: float, clg: float, supply: float, flow: float,
) -> Tuple[float, float, float, float]:
    """Clip to physical bounds and enforce htg <= clg."""
    htg = np.clip(htg, HTG_MIN, HTG_MAX)
    clg = np.clip(clg, CLG_MIN, CLG_MAX)
    if htg > clg:
        mid = (htg + clg) / 2.0
        htg, clg = mid, mid
    supply = np.clip(supply, SUPPLY_MIN, SUPPLY_MAX)
    flow = np.clip(flow, FLOW_MIN, FLOW_MAX)
    return float(htg), float(clg), float(supply), float(flow)


# ── Top 20 BO-optimized parameter sets from nibs_bo_002 (€8,221–€8,463) ────

TOP_20_BO_PARAMS = [
    {"htg_margin": 0.6395, "clg_margin": 1.4321, "night_htg_setback": 0.5962, "supply_temp_low": 18.758, "supply_temp_high": 18.190, "occupied_flow": 0.1054, "unoccupied_flow": 0.0309, "co2_boost_threshold": 432.11, "nightflush_delta": 3.570, "nightflush_flow": 0.2854},
    {"htg_margin": 0.5259, "clg_margin": 1.4314, "night_htg_setback": 0.0185, "supply_temp_low": 18.143, "supply_temp_high": 17.515, "occupied_flow": 0.2296, "unoccupied_flow": 0.0353, "co2_boost_threshold": 415.01, "nightflush_delta": 2.185, "nightflush_flow": 0.3155},
    {"htg_margin": 0.5618, "clg_margin": 1.6510, "night_htg_setback": 0.3963, "supply_temp_low": 18.784, "supply_temp_high": 17.114, "occupied_flow": 0.0651, "unoccupied_flow": 0.0231, "co2_boost_threshold": 425.02, "nightflush_delta": 3.866, "nightflush_flow": 0.2641},
    {"htg_margin": 0.3648, "clg_margin": 1.3805, "night_htg_setback": 0.2612, "supply_temp_low": 18.063, "supply_temp_high": 17.987, "occupied_flow": 0.0806, "unoccupied_flow": 0.0148, "co2_boost_threshold": 403.82, "nightflush_delta": 2.055, "nightflush_flow": 0.2728},
    {"htg_margin": 0.7598, "clg_margin": 1.6921, "night_htg_setback": 0.0421, "supply_temp_low": 18.304, "supply_temp_high": 18.253, "occupied_flow": 0.0899, "unoccupied_flow": 0.0421, "co2_boost_threshold": 473.77, "nightflush_delta": 3.346, "nightflush_flow": 0.2050},
    {"htg_margin": 0.2944, "clg_margin": 1.3495, "night_htg_setback": 0.0343, "supply_temp_low": 17.944, "supply_temp_high": 17.991, "occupied_flow": 0.2619, "unoccupied_flow": 0.0275, "co2_boost_threshold": 445.78, "nightflush_delta": 1.938, "nightflush_flow": 0.2842},
    {"htg_margin": 0.1681, "clg_margin": 1.5235, "night_htg_setback": 0.0114, "supply_temp_low": 18.196, "supply_temp_high": 17.776, "occupied_flow": 0.2180, "unoccupied_flow": 0.0491, "co2_boost_threshold": 474.86, "nightflush_delta": 2.617, "nightflush_flow": 0.3074},
    {"htg_margin": 0.6641, "clg_margin": 1.0657, "night_htg_setback": 0.5818, "supply_temp_low": 20.372, "supply_temp_high": 17.683, "occupied_flow": 0.1396, "unoccupied_flow": 0.0532, "co2_boost_threshold": 421.33, "nightflush_delta": 2.850, "nightflush_flow": 0.2417},
    {"htg_margin": 0.9746, "clg_margin": 1.8029, "night_htg_setback": 0.1335, "supply_temp_low": 18.604, "supply_temp_high": 17.202, "occupied_flow": 0.0728, "unoccupied_flow": 0.0345, "co2_boost_threshold": 421.07, "nightflush_delta": 2.902, "nightflush_flow": 0.2135},
    {"htg_margin": 0.7758, "clg_margin": 1.5055, "night_htg_setback": 0.1324, "supply_temp_low": 18.483, "supply_temp_high": 17.340, "occupied_flow": 0.1712, "unoccupied_flow": 0.0322, "co2_boost_threshold": 408.26, "nightflush_delta": 3.376, "nightflush_flow": 0.3084},
    {"htg_margin": 0.9672, "clg_margin": 1.2693, "night_htg_setback": 0.4718, "supply_temp_low": 18.229, "supply_temp_high": 18.213, "occupied_flow": 0.4356, "unoccupied_flow": 0.0034, "co2_boost_threshold": 427.56, "nightflush_delta": 4.817, "nightflush_flow": 0.2381},
    {"htg_margin": 0.8649, "clg_margin": 1.8515, "night_htg_setback": 0.0971, "supply_temp_low": 18.268, "supply_temp_high": 16.816, "occupied_flow": 0.2228, "unoccupied_flow": 0.0038, "co2_boost_threshold": 411.05, "nightflush_delta": 2.706, "nightflush_flow": 0.2117},
    {"htg_margin": 0.8993, "clg_margin": 1.2021, "night_htg_setback": 0.3336, "supply_temp_low": 20.622, "supply_temp_high": 18.037, "occupied_flow": 0.1776, "unoccupied_flow": 0.0094, "co2_boost_threshold": 464.91, "nightflush_delta": 2.117, "nightflush_flow": 0.2700},
    {"htg_margin": 0.6800, "clg_margin": 1.4147, "night_htg_setback": 0.0115, "supply_temp_low": 17.894, "supply_temp_high": 18.534, "occupied_flow": 0.2940, "unoccupied_flow": 0.0246, "co2_boost_threshold": 419.93, "nightflush_delta": 1.937, "nightflush_flow": 0.3224},
    {"htg_margin": 0.5894, "clg_margin": 1.3317, "night_htg_setback": 0.3530, "supply_temp_low": 19.749, "supply_temp_high": 18.161, "occupied_flow": 0.1115, "unoccupied_flow": 0.0347, "co2_boost_threshold": 425.09, "nightflush_delta": 2.860, "nightflush_flow": 0.2112},
    {"htg_margin": 0.6758, "clg_margin": 1.2924, "night_htg_setback": 0.5505, "supply_temp_low": 20.893, "supply_temp_high": 18.544, "occupied_flow": 0.1031, "unoccupied_flow": 0.0346, "co2_boost_threshold": 440.46, "nightflush_delta": 3.839, "nightflush_flow": 0.3376},
    {"htg_margin": 1.0067, "clg_margin": 1.9207, "night_htg_setback": 0.1200, "supply_temp_low": 18.208, "supply_temp_high": 17.247, "occupied_flow": 0.1858, "unoccupied_flow": 0.0240, "co2_boost_threshold": 422.65, "nightflush_delta": 3.155, "nightflush_flow": 0.2137},
    {"htg_margin": 1.7974, "clg_margin": 0.9527, "night_htg_setback": 1.2259, "supply_temp_low": 20.901, "supply_temp_high": 19.797, "occupied_flow": 0.0531, "unoccupied_flow": 0.0487, "co2_boost_threshold": 439.68, "nightflush_delta": 3.979, "nightflush_flow": 0.3178},
    {"htg_margin": 0.6375, "clg_margin": 0.8213, "night_htg_setback": 0.4534, "supply_temp_low": 18.920, "supply_temp_high": 17.803, "occupied_flow": 0.1237, "unoccupied_flow": 0.0000, "co2_boost_threshold": 445.98, "nightflush_delta": 2.649, "nightflush_flow": 0.2072},
    {"htg_margin": 1.4715, "clg_margin": 1.3140, "night_htg_setback": 1.4420, "supply_temp_low": 18.062, "supply_temp_high": 18.566, "occupied_flow": 0.4509, "unoccupied_flow": 0.0077, "co2_boost_threshold": 443.31, "nightflush_delta": 3.530, "nightflush_flow": 0.4825},
]


class BOParamRBC:
    """Band-tracking controller using a specific BO parameter set.

    seed selects which of the top-20 parameter sets to use (seed % 20).
    """

    def __init__(self, seed: int = 0) -> None:
        idx = seed % len(TOP_20_BO_PARAMS)
        self.p = dict(TOP_20_BO_PARAMS[idx])
        self.p.setdefault("co2_boost_flow", 0.90)
        self._outdoor_buffer: deque = deque(maxlen=RING_BUFFER_SIZE)

    def _update_outdoor(self, outdoor_temp: float) -> float:
        self._outdoor_buffer.append(outdoor_temp)
        return sum(self._outdoor_buffer) / len(self._outdoor_buffer)

    def calculate_setpoints(
        self,
        zone_temp: float,
        outdoor_temp: float,
        return_air_temp: float,
        occupancy: float,
        hour: float,
        day: int,
        co2_concentration: float,
    ) -> Tuple[float, float, float, float]:
        is_occupied = occupancy > 0.5
        p = self.p
        t_out_24h = self._update_outdoor(outdoor_temp)
        lower, upper = _s1_band(t_out_24h)

        htg = lower + p["htg_margin"]
        clg = upper - p["clg_margin"]
        if not is_occupied:
            htg -= p["night_htg_setback"]

        # Supply air temp: linear interp from return air
        lo_ret, hi_ret = 20.0, 26.0
        if return_air_temp <= lo_ret:
            supply = p["supply_temp_low"]
        elif return_air_temp >= hi_ret:
            supply = p["supply_temp_high"]
        else:
            t = (return_air_temp - lo_ret) / (hi_ret - lo_ret)
            supply = p["supply_temp_low"] + t * (p["supply_temp_high"] - p["supply_temp_low"])

        # Fan flow: occupancy + CO2 boost + nightflush
        base_flow = p["occupied_flow"] if is_occupied else p["unoccupied_flow"]
        co2_thr = p["co2_boost_threshold"]
        co2_max = co2_thr + 300.0
        if co2_concentration <= co2_thr:
            co2_flow = 0.0
        elif co2_concentration >= co2_max:
            co2_flow = p["co2_boost_flow"]
        else:
            frac = (co2_concentration - co2_thr) / (co2_max - co2_thr)
            co2_flow = base_flow + frac * (p["co2_boost_flow"] - base_flow)
        flow = max(base_flow, co2_flow)

        if (outdoor_temp < indoor_temp - p["nightflush_delta"]
                and not is_occupied and indoor_temp > 22.0):
            flow = max(flow, p["nightflush_flow"])

        return _clip_actions(htg, clg, supply, flow)


class FullOnRBC:
    """rbc_full_on style: constant setpoints with randomized parameters.

    Randomizes zone setpoint, supply air temp, and fan flow per seed.
    Flow is always substantial (0.3–1.0 kg/s) to keep CO2 reasonable.
    """

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        # Zone setpoint: tight band around 21–23.5°C
        self._zone_setpoint = rng.uniform(21.0, 23.5)
        # Small dead band around setpoint (0–1°C)
        self._dead_band = rng.uniform(0.0, 1.0)
        # Supply air: 17–20°C
        self._supply_temp = rng.uniform(17.0, 20.0)
        # Flow: moderate to high (avoids CO2 blowup from too-low flow)
        self._flow = rng.uniform(0.3, 1.0)

    def calculate_setpoints(
        self,
        zone_temp: float,
        outdoor_temp: float,
        return_air_temp: float,
        occupancy: float,
        hour: float,
        day: int,
        co2_concentration: float,
    ) -> Tuple[float, float, float, float]:
        htg = self._zone_setpoint - self._dead_band / 2.0
        clg = self._zone_setpoint + self._dead_band / 2.0
        return _clip_actions(htg, clg, self._supply_temp, self._flow)


# Registry for easy lookup by name
CONTROLLER_CLASSES = {
    "bo_param": BOParamRBC,
    "full_on": FullOnRBC,
}
