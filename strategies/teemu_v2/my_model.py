"""Improved RBC model based on teemu/alt_model.py.

Changes from alt_model:
1. Rolling window: 24h hourly-updated deque (like original). The scoring
   uses a 6h window but matching it makes setpoints too volatile for the
   building to track. A smooth 24h average keeps zone temps stable.
2. Wider margins: 0.5°C heating / 0.3°C cooling (was 0.03/0.03).
   Building thermal lag causes overshoot past the S1 boundary with
   tiny margins. The scoring counts each 15-min timestep as a full
   hourly penalty (€4/zone/h effective), so even brief excursions
   are extremely expensive.
3. CO2 fan ramp: linear ramp 650→760 ppm replaces 20-step staircase.
4. Supply air: kept original clip(outdoor, 16, 21) logic in neutral zone.
"""

from typing import Tuple
from collections import deque

import numpy as np


# ── S1 band tracker (24h rolling mean, updated hourly like original) ───────

class TemperatureTargeter:
    """Track 24-sample rolling outdoor temp updated once per hour = 24h.

    Using a smoother 24h average keeps setpoints stable so the building
    can actually track them. The scoring's 6h window is volatile but the
    building's thermal mass naturally filters short-term swings.
    """

    def __init__(self):
        self.history = deque(maxlen=24)
        self.t = 0.0

    def update(self, outdoor_temp: float) -> None:
        self.history.append(outdoor_temp)
        self.t = sum(self.history) / len(self.history)

    def get_lower(self) -> float:
        t = self.t
        if len(self.history) < 24:
            return 22.0
        if t <= 0:
            return 20.5
        elif t <= 20:
            return 20.5 + 0.075 * t
        else:
            return 22.0

    def get_upper(self) -> float:
        t = self.t
        if len(self.history) < 24:
            return 25.0
        if t <= 0:
            return 22.0
        elif t <= 15:
            return 22.5 + 0.166 * t
        else:
            return 25.0


# ── Margins — buffer between setpoint and S1 boundary ──────────────────────
# Building thermal lag means zone temp overshoots setpoints. These margins
# keep the zone well inside S1.  Heating margin wider because gas is cheap.

HTG_MARGIN = 0.5   # °C above S1 lower  (gas @ €0.06/kWh)
CLG_MARGIN = 0.3   # °C below S1 upper  (elec @ €0.11/kWh)

# ── CO2 ramp parameters ────────────────────────────────────────────────────

CO2_RAMP_START = 650   # ppm — below this, fan off
CO2_RAMP_FULL = 760    # ppm — at/above this, fan at 1.0 (penalty at 770)


controller = TemperatureTargeter()


class MyModel:
    def __init__(self) -> None:
        self.prev_hour = 0

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

        # ── 1. Update rolling average once per hour (like original) ────────
        if hour != self.prev_hour:
            controller.update(outdoor_temp)
        self.prev_hour = hour

        # ── 2. Zone setpoints with margins inside S1 band ─────────────────
        heating_setpoint = controller.get_lower() + HTG_MARGIN
        cooling_setpoint = max(heating_setpoint, controller.get_upper() - CLG_MARGIN)

        # ── 3. Supply air temp — original logic preserved ──────────────────
        if zone_temp < heating_setpoint + 0.1:
            supply_air_temp = 21.0
        elif zone_temp > cooling_setpoint - 0.1:
            supply_air_temp = 16.0
        else:
            supply_air_temp = float(np.clip(outdoor_temp, 16.0, 21.0))

        # ── 4. CO2 fan control — smooth linear ramp ────────────────────────
        if co2_concentration >= CO2_RAMP_FULL:
            fan_flow_rate = 1.0
        elif co2_concentration <= CO2_RAMP_START:
            fan_flow_rate = 0.0
        else:
            fan_flow_rate = (co2_concentration - CO2_RAMP_START) / (CO2_RAMP_FULL - CO2_RAMP_START)

        return heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow_rate
