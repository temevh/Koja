"""Aggressive energy-minimizing RBC — v4.

Root cause of previous failures:
- v2 (0.5/0.3 margins, 24h hourly): €7,344 energy — margins too wide,
  dead band too narrow, massive over-conditioning.
- v3 (0.03 margins, 6h every-step): €6,865 energy but €12,891 temp
  penalty — 6h window makes setpoints volatile, building can't track.

Samuli's winning recipe: 24-sample deque updated once per hour = 24h
smooth window + 0.03°C margins. The scorer uses rolling(24) on 15-min
data = 6h, but the 24h controller smoothness keeps zones stable inside
S1 despite the window mismatch.

v4 changes to beat Samuli:
1. 96-sample deque updated every 15-min timestep = true 24h, even
   smoother than Samuli's hourly updates. Matches BO model approach.
2. 0.03°C margins (same as Samuli).
3. Samuli's exact 20-step CO2 fan staircase (proven €4 CO2 penalty).
4. Smarter supply air: return-air-compensated (BO best_params insight)
   instead of outdoor clip — avoids heating supply air with gas when
   return air is already warm enough.
5. Unoccupied setback: when CO2 < 420 (empty building), widen dead band
   to save heating/cooling energy with no penalty risk.
"""

from typing import Tuple
from collections import deque

import numpy as np


# ── 24h rolling outdoor temp (96 × 15-min = 24h, updated every step) ──────

class TemperatureTargeter:
    """True 24h rolling mean, updated every timestep for maximum smoothness."""

    def __init__(self):
        self.history = deque(maxlen=96)   # 96 × 15 min = 24 h
        self.t = 0.0

    def update(self, outdoor_temp: float) -> None:
        self.history.append(outdoor_temp)
        self.t = sum(self.history) / len(self.history)

    def get_lower(self) -> float:
        t = self.t
        if len(self.history) < 96:
            return 22.0          # conservative during warmup
        if t <= 0:
            return 20.5
        elif t <= 20:
            return 20.5 + 0.075 * t
        else:
            return 22.0

    def get_upper(self) -> float:
        t = self.t
        if len(self.history) < 96:
            return 25.0          # conservative during warmup
        if t <= 0:
            return 22.0
        elif t <= 15:
            return 22.5 + 0.166 * t
        else:
            return 25.0


# ── Margins ────────────────────────────────────────────────────────────────
HTG_MARGIN = 0.03   # °C above S1 lower
CLG_MARGIN = 0.03   # °C below S1 upper

# ── CO2 thresholds ─────────────────────────────────────────────────────────
CO2_THRESHOLDS = [770, 970, 1220]

# ── Unoccupied detection ───────────────────────────────────────────────────
CO2_EMPTY_THRESHOLD = 420  # ppm — below this, building is likely empty
UNOCCUPIED_SETBACK  = 1.0  # °C wider dead band when empty


controller = TemperatureTargeter()


class MyModel:
    def __init__(self) -> None:
        pass

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

        # ── 1. Update 24h rolling mean every timestep ──────────────────────
        controller.update(outdoor_temp)

        # ── 2. Zone setpoints — tight margins + unoccupied setback ─────────
        lower = controller.get_lower()
        upper = controller.get_upper()

        heating_setpoint = lower + HTG_MARGIN
        cooling_setpoint = max(heating_setpoint, upper - CLG_MARGIN)

        # Widen dead band when building is empty → save energy
        if co2_concentration < CO2_EMPTY_THRESHOLD:
            heating_setpoint -= UNOCCUPIED_SETBACK
            cooling_setpoint += UNOCCUPIED_SETBACK

        # ── 3. Supply air temp — return-air compensated ────────────────────
        # When return air is warm → supply cool air (free or cheap cooling)
        # When return air is cool → supply warm air (gas heating)
        # In neutral zone → avoid unnecessary gas heating
        if zone_temp < heating_setpoint + 0.1:
            supply_air_temp = 21.0          # max heating
        elif zone_temp > cooling_setpoint - 0.1:
            supply_air_temp = 16.0          # max cooling
        else:
            # Neutral: compensate based on return air
            # BO best params: supply_temp_low=18.4 (ret≤20), high=16.8 (ret≥26)
            if return_air_temp <= 20.0:
                supply_air_temp = 18.5
            elif return_air_temp >= 26.0:
                supply_air_temp = 16.8
            else:
                frac = (return_air_temp - 20.0) / 6.0
                supply_air_temp = 18.5 + frac * (16.8 - 18.5)

        # ── 4. CO2 fan control — Samuli's exact staircase ─────────────────
        # Proven to achieve €4 CO2 penalty. Fine-grained ramp prevents
        # both CO2 spikes and excessive fan electricity.
        if co2_concentration >= CO2_THRESHOLDS[0] - 15:     # 755
            fan_flow_rate = 0.94
        elif co2_concentration >= CO2_THRESHOLDS[0] - 25:   # 745
            fan_flow_rate = 0.93
        elif co2_concentration >= CO2_THRESHOLDS[0] - 30:   # 740
            fan_flow_rate = 0.87
        elif co2_concentration >= CO2_THRESHOLDS[0] - 35:   # 735
            fan_flow_rate = 0.80
        elif co2_concentration >= CO2_THRESHOLDS[0] - 40:   # 730
            fan_flow_rate = 0.75
        elif co2_concentration >= CO2_THRESHOLDS[0] - 45:   # 725
            fan_flow_rate = 0.75
        elif co2_concentration >= CO2_THRESHOLDS[0] - 50:   # 720
            fan_flow_rate = 0.70
        elif co2_concentration >= CO2_THRESHOLDS[0] - 55:   # 715
            fan_flow_rate = 0.65
        elif co2_concentration >= CO2_THRESHOLDS[0] - 60:   # 710
            fan_flow_rate = 0.60
        elif co2_concentration >= CO2_THRESHOLDS[0] - 65:   # 705
            fan_flow_rate = 0.55
        elif co2_concentration >= CO2_THRESHOLDS[0] - 70:   # 700
            fan_flow_rate = 0.50
        elif co2_concentration >= CO2_THRESHOLDS[0] - 75:   # 695
            fan_flow_rate = 0.45
        elif co2_concentration >= CO2_THRESHOLDS[0] - 80:   # 690
            fan_flow_rate = 0.40
        elif co2_concentration >= CO2_THRESHOLDS[0] - 85:   # 685
            fan_flow_rate = 0.35
        elif co2_concentration >= CO2_THRESHOLDS[0] - 90:   # 680
            fan_flow_rate = 0.30
        elif co2_concentration >= CO2_THRESHOLDS[0] - 95:   # 675
            fan_flow_rate = 0.25
        elif co2_concentration >= CO2_THRESHOLDS[0] - 100:  # 670
            fan_flow_rate = 0.20
        elif co2_concentration >= CO2_THRESHOLDS[0] - 110:  # 660
            fan_flow_rate = 0.15
        elif co2_concentration >= CO2_THRESHOLDS[0] - 120:  # 650
            fan_flow_rate = 0.10
        elif co2_concentration >= CO2_THRESHOLDS[0] - 250:  # 520
            fan_flow_rate = 0.05
        else:
            fan_flow_rate = 0.00

        return heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow_rate
