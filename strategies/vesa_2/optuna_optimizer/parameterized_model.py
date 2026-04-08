from dataclasses import dataclass
from typing import Tuple
import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _ensure_high_from_low(low: float, high: float, min_gap: float = 0.0) -> Tuple[float, float]:
    low = float(low)
    high = float(high)
    if high < low + min_gap:
        high = low + min_gap
    return low, high


@dataclass
class AHUModeParams:
    flow: float
    mode_name: str

class ParameterizedRBCModel:
    def __init__(
        self,
        zone_htg_setpoint_low: float = 21.0,
        zone_htg_setpoint_high: float = 22.0,
        zone_clg_setpoint_low: float = 23.0,
        zone_clg_setpoint_high: float = 25.0,
        outdoor_temp_low: float = 0.0,
        outdoor_temp_high: float = 20.0,
        return_air_temp_low: float = 21.5,
        return_air_temp_high: float = 24.5,
        sup_temp_at_low: float = 19.0,
        sup_temp_at_high: float = 19.0,
        co2_min_limit: float = 600.0,
        co2_max_limit: float = 800.0,
        outdoor_temp_low_limit: float = -25.0,
        outdoor_temp_high_limit: float = -15.0,
        flow_low: float = 0.15,
        flow_moderate: float = 0.3,
        flow_boost: float = 1.0,
        flow_max: float = 1.0,
    ) -> None:
        htg_low, htg_high = _ensure_high_from_low(zone_htg_setpoint_low, zone_htg_setpoint_high)
        clg_low, clg_high = _ensure_high_from_low(zone_clg_setpoint_low, zone_clg_setpoint_high)
        otemp_low, otemp_high = _ensure_high_from_low(outdoor_temp_low, outdoor_temp_high, min_gap=0.1)
        rat_low, rat_high = _ensure_high_from_low(return_air_temp_low, return_air_temp_high, min_gap=0.1)
        co2_low, co2_high = _ensure_high_from_low(co2_min_limit, co2_max_limit, min_gap=1.0)
        limit_low, limit_high = _ensure_high_from_low(outdoor_temp_low_limit, outdoor_temp_high_limit, min_gap=0.1)

        # Keep 1.0 C minimum deadband to avoid invalid thermostat ranges in E+.
        if clg_low < htg_high + 1.0:
            clg_low = htg_high + 1.0
        if clg_high < clg_low:
            clg_high = clg_low

        flow_low = _clamp(flow_low, 0.0, 1.0)
        flow_moderate = _clamp(flow_moderate, flow_low, 1.0)
        flow_boost = _clamp(flow_boost, flow_moderate, 1.0)
        flow_max = _clamp(flow_max, flow_boost, 1.0)

        self.ZONE_HTG_SETPOINT_LOW = _clamp(htg_low, 18.0, 25.0)
        self.ZONE_HTG_SETPOINT_HIGH = _clamp(htg_high, self.ZONE_HTG_SETPOINT_LOW, 25.0)
        self.ZONE_CLG_SETPOINT_LOW = _clamp(clg_low, 18.0, 25.0)
        self.ZONE_CLG_SETPOINT_HIGH = _clamp(clg_high, self.ZONE_CLG_SETPOINT_LOW, 25.0)
        self.OUTDOOR_TEMP_LOW = float(otemp_low)
        self.OUTDOOR_TEMP_HIGH = float(otemp_high)
        self.RETURN_AIR_TEMP_LOW = float(rat_low)
        self.RETURN_AIR_TEMP_HIGH = float(rat_high)
        self.SUP_TEMP_AT_LOW = float(sup_temp_at_low)
        self.SUP_TEMP_AT_HIGH = float(sup_temp_at_high)
        self.CO2_MIN_LIMIT = float(co2_low)
        self.CO2_MAX_LIMIT = float(co2_high)
        self.OUTDOOR_TEMP_LOW_LIMIT = float(limit_low)
        self.OUTDOOR_TEMP_HIGH_LIMIT = float(limit_high)
        self.FLOW_LOW = float(flow_low)
        self.FLOW_MODERATE = float(flow_moderate)
        self.FLOW_BOOST = float(flow_boost)
        self.FLOW_MAX = float(flow_max)
        
        self.htg_setpoint: float = self.ZONE_HTG_SETPOINT_LOW
        self.clg_setpoint: float = self.ZONE_CLG_SETPOINT_LOW
        self._co2_boosting: bool = False
        self.mode_params = {
            0: AHUModeParams(flow=0.0, mode_name="off"),
            1: AHUModeParams(flow=self.FLOW_LOW, mode_name="low"),
            2: AHUModeParams(flow=self.FLOW_MODERATE, mode_name="moderate"),
            3: AHUModeParams(flow=self.FLOW_BOOST, mode_name="boost"),
        }

    def co2_flow_control(self, hour: float, day: int, co2_concentration: float, outdoortemp: float) -> float:
        is_workday = day != 1 and day != 7
        is_nightflush = is_workday and 1.5 <= hour < 2.5 
        is_nightflush_weekend = not is_workday and 0.0 <= hour < 2.0 
        is_before_work_flush = is_workday and 4.5 <= hour < 5.5 
        is_weekend_flush = not is_workday and 6.0 <= hour < 17.0 
        is_working_hours = is_workday and 5.5 <= hour < 23.5 

        base_flow = 0.0
        if is_before_work_flush:
            base_flow = self.FLOW_MODERATE
        elif is_nightflush or is_nightflush_weekend or is_weekend_flush or is_working_hours:
            base_flow = self.FLOW_LOW
        else:
            base_flow = 0.0

        co2_flow = 0.0
        if co2_concentration <= self.CO2_MIN_LIMIT:
            co2_flow = 0.0
        elif co2_concentration >= self.CO2_MAX_LIMIT:
            co2_flow = self.FLOW_BOOST
        else:
            fraction = (co2_concentration - self.CO2_MIN_LIMIT) / (self.CO2_MAX_LIMIT - self.CO2_MIN_LIMIT)
            co2_flow = self.FLOW_LOW + fraction * (self.FLOW_BOOST - self.FLOW_LOW)

        target_flow = max(co2_flow, base_flow)  

        max_allowed = target_flow  
        if outdoortemp <= self.OUTDOOR_TEMP_LOW_LIMIT:
            max_allowed = self.FLOW_LOW
        elif outdoortemp >= self.OUTDOOR_TEMP_HIGH_LIMIT:
            max_allowed = self.FLOW_MAX
        else:
            slope = (self.FLOW_MAX - self.FLOW_LOW) / (self.OUTDOOR_TEMP_HIGH_LIMIT - self.OUTDOOR_TEMP_LOW_LIMIT)
            max_allowed = self.FLOW_LOW + slope * (outdoortemp - self.OUTDOOR_TEMP_LOW_LIMIT)

        return min(target_flow, max_allowed)

    def return_air_compensation(self, return_air_temp: float) -> float:
        if return_air_temp <= self.RETURN_AIR_TEMP_LOW:
            return self.SUP_TEMP_AT_LOW
        if return_air_temp >= self.RETURN_AIR_TEMP_HIGH:
            return self.SUP_TEMP_AT_HIGH

        slope = (self.SUP_TEMP_AT_HIGH - self.SUP_TEMP_AT_LOW) / (self.RETURN_AIR_TEMP_HIGH - self.RETURN_AIR_TEMP_LOW)
        return float(np.clip(
            self.SUP_TEMP_AT_LOW + slope * (return_air_temp - self.RETURN_AIR_TEMP_LOW),
            min(self.SUP_TEMP_AT_LOW, self.SUP_TEMP_AT_HIGH),
            max(self.SUP_TEMP_AT_LOW, self.SUP_TEMP_AT_HIGH)
        ))

    def zone_setpoints(self, outdoor_temp: float) -> Tuple[float, float]:
        t = np.clip((outdoor_temp - self.OUTDOOR_TEMP_LOW) / (self.OUTDOOR_TEMP_HIGH - self.OUTDOOR_TEMP_LOW), 0.0, 1.0)
        htg = self.ZONE_HTG_SETPOINT_LOW + t * (self.ZONE_HTG_SETPOINT_HIGH - self.ZONE_HTG_SETPOINT_LOW)
        clg = self.ZONE_CLG_SETPOINT_LOW + t * (self.ZONE_CLG_SETPOINT_HIGH - self.ZONE_CLG_SETPOINT_LOW)
        if clg < htg + 1.0:
            clg = htg + 1.0
        htg = float(np.clip(htg, 18.0, 24.0))
        clg = float(np.clip(clg, htg + 0.5, 25.0))
        return float(htg), float(clg)
    
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
        supply_air_temp = self.return_air_compensation(return_air_temp)
        htg, clg = self.zone_setpoints(outdoor_temp)

        # Slight night setback for heating and setup for cooling
        if hour < 5.5 or hour >= 20.0:
            htg = max(18.0, htg - 0.5)
            clg = min(25.0, clg + 0.5)

        self.htg_setpoint = htg
        self.clg_setpoint = clg

        flow = self.co2_flow_control(hour, day, co2_concentration, outdoor_temp)

        return self.htg_setpoint, self.clg_setpoint, supply_air_temp, flow
