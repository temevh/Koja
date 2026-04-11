"""MPC controller model — main interface for EnergyPlus.

Implements calculate_setpoints() as required by the controller template.
Uses direct S1 comfort-band tracking for zone setpoints (htg/clg) and
reactive sensor-driven control for supply air temperature and fan flow.

Also supports update_state() to receive full per-zone observations from
the extended energyplus_controller.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_STRATEGY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRATEGY_DIR))

RING_BUFFER_SIZE = 96  # 24h at 15-min timesteps

# Tuned parameters (from Bayesian Optimization best params)
HTG_MARGIN = 1.19          # °C above S1 lower → heating setpoint
CLG_MARGIN = 1.27          # °C below S1 upper → cooling setpoint
NIGHT_HTG_SETBACK = 0.96   # °C additional margin when unoccupied

SUPPLY_TEMP_LOW = 18.4     # supply temp when return air is cool
SUPPLY_TEMP_HIGH = 16.8    # supply temp when return air is warm

OCCUPIED_FLOW = 0.40       # base flow when occupied [kg/s]
UNOCCUPIED_FLOW = 0.11     # flow when building is empty [kg/s]

CO2_BOOST_THRESHOLD = 415.0  # ppm to start ramping up flow
CO2_BOOST_FLOW = 0.90        # max flow at high CO2 [kg/s]
CO2_BOOST_RANGE = 300.0      # ppm above threshold for full boost

NIGHTFLUSH_DELTA = 3.65    # °C: enable when outdoor < indoor - delta
NIGHTFLUSH_FLOW = 0.43     # flow during free cooling [kg/s]


class MPCModel:
    """S1 band-tracking controller with reactive flow and supply temp."""

    def __init__(self, **kwargs) -> None:
        # Accept but ignore CEM/dynamics kwargs for backward compatibility
        self._outdoor_buffer: deque = deque(maxlen=RING_BUFFER_SIZE)
        self._obs: Optional[dict] = None
        self._step = 0

    # -- 24h rolling outdoor temperature mean --------------------------------

    def _update_outdoor_rolling(self, outdoor_temp: float) -> float:
        self._outdoor_buffer.append(outdoor_temp)
        return sum(self._outdoor_buffer) / len(self._outdoor_buffer)

    # -- S1 comfort band (matches scoring.py exactly) ------------------------

    @staticmethod
    def _s1_band(t_out_24h: float) -> Tuple[float, float]:
        if t_out_24h <= 0:
            lower = 20.5
        elif t_out_24h <= 20:
            lower = 20.5 + 0.075 * t_out_24h
        else:
            lower = 22.0

        if t_out_24h <= 0:
            upper = 22.0
        elif t_out_24h <= 15:
            upper = 22.5 + 0.166 * t_out_24h
        else:
            upper = 25.0

        return lower, upper

    # -- Zone setpoints (band-tracking + night setback) ----------------------

    def _zone_setpoints(
        self, t_out_24h: float, is_occupied: bool,
    ) -> Tuple[float, float]:
        lower, upper = self._s1_band(t_out_24h)

        htg = lower + HTG_MARGIN
        clg = upper - CLG_MARGIN

        if not is_occupied:
            htg -= NIGHT_HTG_SETBACK

        # Safety: htg can't exceed clg
        if htg > clg:
            mid = (htg + clg) / 2.0
            htg, clg = mid, mid

        return float(htg), float(clg)

    # -- Supply air temperature (return-air-compensated) ---------------------

    @staticmethod
    def _supply_temp(return_air_temp: float) -> float:
        lo_ret, hi_ret = 20.0, 26.0
        if return_air_temp <= lo_ret:
            return SUPPLY_TEMP_LOW
        if return_air_temp >= hi_ret:
            return SUPPLY_TEMP_HIGH

        t = (return_air_temp - lo_ret) / (hi_ret - lo_ret)
        val = SUPPLY_TEMP_LOW + t * (SUPPLY_TEMP_HIGH - SUPPLY_TEMP_LOW)
        return float(np.clip(val, min(SUPPLY_TEMP_LOW, SUPPLY_TEMP_HIGH),
                             max(SUPPLY_TEMP_LOW, SUPPLY_TEMP_HIGH)))

    # -- Fan flow (reactive: occupancy + CO2 + free cooling) ----------------

    @staticmethod
    def _fan_flow(
        is_occupied: bool,
        co2_concentration: float,
        outdoor_temp: float,
        indoor_temp: float,
    ) -> float:
        base_flow = OCCUPIED_FLOW if is_occupied else UNOCCUPIED_FLOW

        # CO2 demand boost
        if co2_concentration <= CO2_BOOST_THRESHOLD:
            co2_flow = 0.0
        elif co2_concentration >= CO2_BOOST_THRESHOLD + CO2_BOOST_RANGE:
            co2_flow = CO2_BOOST_FLOW
        else:
            frac = (co2_concentration - CO2_BOOST_THRESHOLD) / CO2_BOOST_RANGE
            co2_flow = base_flow + frac * (CO2_BOOST_FLOW - base_flow)

        target_flow = max(base_flow, co2_flow)

        # Free cooling nightflush
        if (outdoor_temp < indoor_temp - NIGHTFLUSH_DELTA
                and not is_occupied
                and indoor_temp > 22.0):
            target_flow = max(target_flow, NIGHTFLUSH_FLOW)

        return float(np.clip(target_flow, 0.0, 1.0))

    # -- Interface -----------------------------------------------------------

    def update_state(self, obs: dict) -> None:
        """Receive full per-zone observations from the controller."""
        self._obs = obs

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
        """Compute setpoints via S1 band tracking + reactive flow."""
        t_out_24h = self._update_outdoor_rolling(outdoor_temp)
        self._step += 1

        is_occupied = occupancy > 0.5

        htg, clg = self._zone_setpoints(t_out_24h, is_occupied)
        supply = self._supply_temp(return_air_temp)
        flow = self._fan_flow(is_occupied, co2_concentration,
                              outdoor_temp, zone_temp)

        if self._step % 96 == 0:  # Log once per day
            lower, upper = self._s1_band(t_out_24h)
            print(f"  MPC step {self._step}: htg={htg:.1f} clg={clg:.1f} "
                  f"supply={supply:.1f} flow={flow:.2f} "
                  f"S1=[{lower:.1f},{upper:.1f}] t_out_24h={t_out_24h:.1f}°C")

        return (float(htg), float(clg), float(supply), float(flow))
