from dataclasses import dataclass
from collections import deque
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
        precondition_hours: float = 1.0,
        work_start: float = 6.0,
        work_end: float = 18.0,
        extended_end: float = 22.0,
        night_htg_setback: float = 0.2,
        night_clg_setup: float = 0.2,
        min_deadband: float = 1.0,
        flow_night: float = 0.05,
        flow_pre_flush: float = 0.35,
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
        precondition_hours = _clamp(precondition_hours, 0.0, 4.0)
        work_start = _clamp(work_start, 0.0, 12.0)
        work_end = _clamp(work_end, max(work_start + 4.0, 8.0), 24.0)
        extended_end = _clamp(extended_end, work_end, 24.0)
        night_htg_setback = _clamp(night_htg_setback, 0.0, 2.0)
        night_clg_setup = _clamp(night_clg_setup, 0.0, 2.0)
        min_deadband = _clamp(min_deadband, 0.5, 3.0)
        flow_night = _clamp(flow_night, 0.0, flow_low)
        flow_pre_flush = _clamp(flow_pre_flush, flow_low, flow_boost)

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
        self.PRECONDITION_HOURS = float(precondition_hours)
        self.WORK_START = float(work_start)
        self.WORK_END = float(work_end)
        self.EXTENDED_END = float(extended_end)
        self.NIGHT_HTG_SETBACK = float(night_htg_setback)
        self.NIGHT_CLG_SETUP = float(night_clg_setup)
        self.MIN_DEADBAND = float(min_deadband)
        self.FLOW_NIGHT = float(flow_night)
        self.FLOW_PRE_FLUSH = float(flow_pre_flush)
        
        self.htg_setpoint: float = self.ZONE_HTG_SETPOINT_LOW
        self.clg_setpoint: float = self.ZONE_CLG_SETPOINT_LOW
        self._co2_boosting: bool = False
        self._outdoor_history: deque = deque(maxlen=96)  # 24h at 15-min timestep
        self.mode_params = {
            0: AHUModeParams(flow=0.0, mode_name="off"),
            1: AHUModeParams(flow=self.FLOW_LOW, mode_name="low"),
            2: AHUModeParams(flow=self.FLOW_MODERATE, mode_name="moderate"),
            3: AHUModeParams(flow=self.FLOW_BOOST, mode_name="boost"),
        }

    def effective_params(self) -> dict:
        return {
            "zone_htg_setpoint_low": self.ZONE_HTG_SETPOINT_LOW,
            "zone_htg_setpoint_high": self.ZONE_HTG_SETPOINT_HIGH,
            "zone_clg_setpoint_low": self.ZONE_CLG_SETPOINT_LOW,
            "zone_clg_setpoint_high": self.ZONE_CLG_SETPOINT_HIGH,
            "outdoor_temp_low": self.OUTDOOR_TEMP_LOW,
            "outdoor_temp_high": self.OUTDOOR_TEMP_HIGH,
            "return_air_temp_low": self.RETURN_AIR_TEMP_LOW,
            "return_air_temp_high": self.RETURN_AIR_TEMP_HIGH,
            "sup_temp_at_low": self.SUP_TEMP_AT_LOW,
            "sup_temp_at_high": self.SUP_TEMP_AT_HIGH,
            "co2_min_limit": self.CO2_MIN_LIMIT,
            "co2_max_limit": self.CO2_MAX_LIMIT,
            "outdoor_temp_low_limit": self.OUTDOOR_TEMP_LOW_LIMIT,
            "outdoor_temp_high_limit": self.OUTDOOR_TEMP_HIGH_LIMIT,
            "flow_low": self.FLOW_LOW,
            "flow_moderate": self.FLOW_MODERATE,
            "flow_boost": self.FLOW_BOOST,
            "flow_max": self.FLOW_MAX,
            "precondition_hours": self.PRECONDITION_HOURS,
            "work_start": self.WORK_START,
            "work_end": self.WORK_END,
            "extended_end": self.EXTENDED_END,
            "night_htg_setback": self.NIGHT_HTG_SETBACK,
            "night_clg_setup": self.NIGHT_CLG_SETUP,
            "min_deadband": self.MIN_DEADBAND,
            "flow_night": self.FLOW_NIGHT,
            "flow_pre_flush": self.FLOW_PRE_FLUSH,
        }

    def _outdoor_24h_mean(self, outdoor_temp: float) -> float:
        self._outdoor_history.append(float(outdoor_temp))
        return float(sum(self._outdoor_history) / len(self._outdoor_history))

    def _s1_lower(self, t_mean: float) -> float:
        if t_mean <= 0.0:
            return 20.5
        if t_mean <= 20.0:
            return 20.5 + 0.075 * t_mean
        return 22.0

    def _s1_upper(self, t_mean: float) -> float:
        if t_mean <= 0.0:
            return 22.0
        if t_mean <= 15.0:
            return 22.0 + 0.2 * t_mean
        return 25.0

    def _s2_lower(self, t_mean: float) -> float:
        if t_mean <= 0.0:
            return 20.5
        if t_mean <= 20.0:
            return 20.5 + 0.025 * t_mean
        return 21.0

    def co2_flow_control(self, hour: float, day: int, co2_concentration: float, outdoortemp: float) -> float:
        is_workday = day != 1 and day != 7
        is_pre_condition = is_workday and (self.WORK_START - self.PRECONDITION_HOURS) <= hour < self.WORK_START
        is_working_hours = is_workday and self.WORK_START <= hour < self.WORK_END
        is_extended = is_workday and self.WORK_END <= hour < self.EXTENDED_END

        if is_pre_condition:
            base_flow = self.FLOW_PRE_FLUSH
        elif is_working_hours or is_extended:
            base_flow = self.FLOW_LOW
        else:
            base_flow = self.FLOW_NIGHT

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
        t_mean = self._outdoor_24h_mean(outdoor_temp)
        s1_low = self._s1_lower(t_mean)
        s1_up = self._s1_upper(t_mean)

        # Map optimized absolute anchors to dynamic S1-relative margins.
        # At cold side, S1 is narrower/lower than warm side.
        h_margin_cold = self.ZONE_HTG_SETPOINT_LOW - 20.5
        h_margin_warm = self.ZONE_HTG_SETPOINT_HIGH - 22.0
        c_margin_cold = 22.0 - self.ZONE_CLG_SETPOINT_LOW
        c_margin_warm = 25.0 - self.ZONE_CLG_SETPOINT_HIGH

        blend = float(np.clip((t_mean - 0.0) / 20.0, 0.0, 1.0))
        h_margin = h_margin_cold + blend * (h_margin_warm - h_margin_cold)
        c_margin = c_margin_cold + blend * (c_margin_warm - c_margin_cold)

        htg = s1_low + h_margin
        clg = s1_up - c_margin
        if clg < htg + 1.0:
            clg = htg + 1.0
        htg = float(np.clip(htg, 19.0, 24.0))
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
        is_workday = day != 1 and day != 7
        is_pre_condition = is_workday and (self.WORK_START - self.PRECONDITION_HOURS) <= hour < self.WORK_START
        is_working = is_workday and self.WORK_START <= hour < self.WORK_END
        is_extended = is_workday and self.WORK_END <= hour < self.EXTENDED_END

        htg, clg = self.zone_setpoints(outdoor_temp)

        if not (is_pre_condition or is_working or is_extended):
            # Mild relaxation only; large setbacks created heavy temp penalties.
            s2_low = self._s2_lower(self._outdoor_24h_mean(outdoor_temp))
            htg = max(s2_low + 0.2, htg - self.NIGHT_HTG_SETBACK)
            clg = min(25.0, clg + self.NIGHT_CLG_SETUP)

        if clg - htg < self.MIN_DEADBAND:
            mid = 0.5 * (clg + htg)
            htg = mid - 0.5 * self.MIN_DEADBAND
            clg = mid + 0.5 * self.MIN_DEADBAND
        htg = float(np.clip(htg, 19.0, 24.0))
        clg = float(np.clip(clg, htg + 0.5, 25.0))

        supply_air_temp = self.return_air_compensation(return_air_temp)
        # Add small feedback trim to improve comfort recovery during cold/hot excursions.
        if zone_temp < htg - 0.3:
            supply_air_temp = min(21.0, supply_air_temp + 0.8)
        elif zone_temp > clg + 0.3:
            supply_air_temp = max(16.0, supply_air_temp - 0.6)

        self.htg_setpoint = htg
        self.clg_setpoint = clg

        flow = self.co2_flow_control(hour, day, co2_concentration, outdoor_temp)

        return self.htg_setpoint, self.clg_setpoint, supply_air_temp, flow
