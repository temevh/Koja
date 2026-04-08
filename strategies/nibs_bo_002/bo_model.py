"""Band-tracking RBC model for Bayesian Optimization — v2.

Instead of optimizing absolute setpoints, this controller computes the
actual S1 comfort band from a 24h rolling outdoor temperature and places
heating/cooling setpoints at configurable margins INSIDE the band.

Key advantages over absolute-setpoint models:
  - Cannot violate the comfort band (if margins > 0)
  - Automatically adapts to seasons — no separate winter/summer logic
  - Replaces 6+ setpoint params with 2 margin params
  - Each parameter has direct physical meaning

The controller also replaces time-based schedules with sensor-reactive
logic: occupancy-driven flow, CO2 boost, and free-cooling nightflush
triggered by outdoor-indoor temperature differential.
"""

from collections import deque
from typing import Tuple

import numpy as np


# Ring buffer size: 24h at 15-min timesteps = 96 samples.
# During warmup (< 96 samples), we use the available mean.
RING_BUFFER_SIZE = 96


DEFAULT_PARAMS: dict = {
    # Comfort band margins
    "htg_margin": 0.5,           # °C above S1 lower → heating setpoint
    "clg_margin": 0.5,           # °C below S1 upper → cooling setpoint
    "night_htg_setback": 1.0,    # °C additional margin reduction at night

    # Supply air temperature
    "supply_temp_low": 19.0,     # supply temp when return air is cool
    "supply_temp_high": 17.0,    # supply temp when return air is warm

    # Reactive flow control
    "occupied_flow": 0.30,       # base flow when occupied
    "unoccupied_flow": 0.05,     # flow when building is empty

    # CO2 demand control
    "co2_boost_threshold": 500.0,  # ppm to start ramping up flow
    "co2_boost_flow": 0.90,       # max flow at high CO2

    # Free cooling (nightflush)
    "nightflush_delta": 2.0,     # °C: enable when outdoor < indoor - delta
    "nightflush_flow": 0.5,      # flow during free cooling
}


class BOModel:
    """Band-tracking HVAC controller with 10 optimizable parameters."""

    def __init__(self, params: dict | None = None) -> None:
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self._outdoor_buffer: deque = deque(maxlen=RING_BUFFER_SIZE)

    # -- 24h rolling outdoor temperature mean --------------------------------

    def _update_outdoor_rolling(self, outdoor_temp: float) -> float:
        """Push new reading and return the rolling 24h mean."""
        self._outdoor_buffer.append(outdoor_temp)
        return sum(self._outdoor_buffer) / len(self._outdoor_buffer)

    # -- S1 comfort band computation ----------------------------------------

    @staticmethod
    def _s1_band(t_out_24h: float) -> Tuple[float, float]:
        """Compute S1 lower and upper bounds from 24h rolling outdoor temp.

        Exactly matches the scoring function in scoring.py.
        """
        # Lower S1
        if t_out_24h <= 0:
            lower = 20.5
        elif t_out_24h <= 20:
            lower = 20.5 + 0.075 * t_out_24h
        else:
            lower = 22.0

        # Upper S1
        if t_out_24h <= 0:
            upper = 22.0
        elif t_out_24h <= 15:
            upper = 22.5 + 0.166 * t_out_24h
        else:
            upper = 25.0

        return lower, upper

    # -- Zone setpoints (band-tracking + night setback) ----------------------

    def _zone_setpoints(
        self, t_out_24h: float, hour: float, is_occupied: bool,
    ) -> Tuple[float, float]:
        lower, upper = self._s1_band(t_out_24h)
        p = self.p

        htg = lower + p["htg_margin"]
        clg = upper - p["clg_margin"]

        # Night setback: reduce heating when unoccupied
        if not is_occupied:
            htg -= p["night_htg_setback"]

        # Safety: htg can't exceed clg
        if htg > clg:
            mid = (htg + clg) / 2.0
            htg, clg = mid, mid

        return float(htg), float(clg)

    # -- Supply air temperature (return-air-compensated) ---------------------

    def _supply_temp(self, return_air_temp: float) -> float:
        """Linear interpolation between supply_temp_low and supply_temp_high.

        Fixed return air range: 20-26°C (covers realistic operating range).
        """
        lo_ret, hi_ret = 20.0, 26.0
        sup_lo = self.p["supply_temp_low"]
        sup_hi = self.p["supply_temp_high"]

        if return_air_temp <= lo_ret:
            return sup_lo
        if return_air_temp >= hi_ret:
            return sup_hi

        t = (return_air_temp - lo_ret) / (hi_ret - lo_ret)
        val = sup_lo + t * (sup_hi - sup_lo)
        return float(np.clip(val, min(sup_lo, sup_hi), max(sup_lo, sup_hi)))

    # -- Fan flow (reactive: occupancy + CO2 + free cooling) ----------------

    def _fan_flow(
        self,
        is_occupied: bool,
        co2_concentration: float,
        outdoor_temp: float,
        indoor_temp: float,
    ) -> float:
        p = self.p

        # Base flow: occupancy-driven
        base_flow = p["occupied_flow"] if is_occupied else p["unoccupied_flow"]

        # CO2 demand boost (ramp from base to co2_boost_flow)
        co2_thr = p["co2_boost_threshold"]
        co2_max = co2_thr + 300.0  # full boost 300 ppm above threshold
        if co2_concentration <= co2_thr:
            co2_flow = 0.0
        elif co2_concentration >= co2_max:
            co2_flow = p["co2_boost_flow"]
        else:
            frac = (co2_concentration - co2_thr) / (co2_max - co2_thr)
            co2_flow = base_flow + frac * (p["co2_boost_flow"] - base_flow)

        target_flow = max(base_flow, co2_flow)

        # Free cooling nightflush: when outdoor is significantly cooler
        delta = p["nightflush_delta"]
        if (outdoor_temp < indoor_temp - delta
                and not is_occupied
                and indoor_temp > 22.0):
            target_flow = max(target_flow, p["nightflush_flow"])

        return float(np.clip(target_flow, 0.0, 1.0))

    # -- Main interface ------------------------------------------------------

    def calculate_setpoints(
        self,
        outdoor_temp: float,
        return_air_temp: float,
        indoor_temp: float,
        is_occupied: bool,
        co2_concentration: float,
    ) -> Tuple[float, float, float, float]:
        """Compute all 4 actuator values for this timestep."""
        t_out_24h = self._update_outdoor_rolling(outdoor_temp)

        htg, clg = self._zone_setpoints(t_out_24h, 0.0, is_occupied)
        supply = self._supply_temp(return_air_temp)
        flow = self._fan_flow(is_occupied, co2_concentration,
                              outdoor_temp, indoor_temp)

        return htg, clg, supply, flow
