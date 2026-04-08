"""Vesa5 adaptive rule-based HVAC controller."""

from collections import deque
from typing import Tuple


class Vesa5Model:
    def __init__(
        self,
        htg_margin: float = 0.35,
        clg_margin: float = 0.35,
        night_htg_setback: float = 0.9,
        night_clg_setup: float = 0.9,
        precondition_hours: float = 1.0,
        sup_temp_cold: float = 19.0,
        sup_temp_warm: float = 17.5,
        return_air_cold_ref: float = 21.5,
        return_air_warm_ref: float = 24.5,
        co2_low_threshold: float = 600.0,
        co2_high_threshold: float = 780.0,
        co2_emergency: float = 920.0,
        flow_min_occupied: float = 0.16,
        flow_boost: float = 1.0,
        flow_night: float = 0.0,
        flow_pre_flush: float = 0.40,
        cold_limit_low: float = -25.0,
        cold_limit_high: float = -15.0,
        work_start: float = 6.0,
        work_end: float = 18.0,
        extended_end: float = 22.0,
        min_deadband: float = 1.0,
    ) -> None:
        self.htg_margin = htg_margin
        self.clg_margin = clg_margin
        self.night_htg_setback = night_htg_setback
        self.night_clg_setup = night_clg_setup
        self.precondition_hours = precondition_hours
        self.sup_temp_cold = sup_temp_cold
        self.sup_temp_warm = sup_temp_warm
        self.return_air_cold_ref = return_air_cold_ref
        self.return_air_warm_ref = return_air_warm_ref
        self.co2_low_threshold = co2_low_threshold
        self.co2_high_threshold = co2_high_threshold
        self.co2_emergency = co2_emergency
        self.flow_min_occupied = flow_min_occupied
        self.flow_boost = flow_boost
        self.flow_night = flow_night
        self.flow_pre_flush = flow_pre_flush
        self.cold_limit_low = cold_limit_low
        self.cold_limit_high = cold_limit_high
        self.work_start = work_start
        self.work_end = work_end
        self.extended_end = extended_end
        self.min_deadband = min_deadband

        self._outdoor_history: deque = deque(maxlen=96)  # 24h at 15 min
        self._co2_high_latched = False

    def _outdoor_24h_mean(self, outdoor_temp: float) -> float:
        self._outdoor_history.append(float(outdoor_temp))
        return sum(self._outdoor_history) / len(self._outdoor_history)

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

    def _return_air_compensation(self, return_air_temp: float) -> float:
        if return_air_temp <= self.return_air_cold_ref:
            return self.sup_temp_cold
        if return_air_temp >= self.return_air_warm_ref:
            return self.sup_temp_warm
        span = self.return_air_warm_ref - self.return_air_cold_ref
        ratio = (return_air_temp - self.return_air_cold_ref) / max(span, 0.001)
        return self.sup_temp_cold + ratio * (self.sup_temp_warm - self.sup_temp_cold)

    def _co2_flow(self, co2: float) -> float:
        if co2 >= self.co2_emergency:
            self._co2_high_latched = True
            return self.flow_boost
        if co2 <= self.co2_low_threshold:
            self._co2_high_latched = False
            return 0.0
        if co2 >= self.co2_high_threshold:
            self._co2_high_latched = True
            return self.flow_boost

        ratio = (co2 - self.co2_low_threshold) / max(
            self.co2_high_threshold - self.co2_low_threshold, 1.0
        )
        flow = self.flow_min_occupied + ratio * (self.flow_boost - self.flow_min_occupied)
        if self._co2_high_latched and co2 > self.co2_low_threshold + 30.0:
            return max(flow, self.flow_min_occupied)
        return flow

    def _cold_weather_limit(self, flow: float, outdoor_temp: float) -> float:
        """Linear cap from flow_min_occupied at cold_limit_low to full flow at cold_limit_high."""
        if outdoor_temp >= self.cold_limit_high:
            return flow
        if outdoor_temp <= self.cold_limit_low:
            return min(flow, self.flow_min_occupied)

        ratio = (outdoor_temp - self.cold_limit_low) / max(
            self.cold_limit_high - self.cold_limit_low, 0.001
        )
        allowed_max = self.flow_min_occupied + ratio * (1.0 - self.flow_min_occupied)
        return min(flow, allowed_max)

    def calculate_setpoints(
        self,
        zone_temp: float,
        outdoor_temp: float,
        return_air_temp: float,
        occupancy: float,
        hour: float,
        day: int,
        co2_concentration: float,
        month: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        del zone_temp, month  # reserved for future features

        t_mean = self._outdoor_24h_mean(outdoor_temp)
        s1_low = self._s1_lower(t_mean)
        s1_high = self._s1_upper(t_mean)
        s2_low = self._s2_lower(t_mean)

        is_workday = day not in (1, 7)
        is_working = is_workday and self.work_start <= hour < self.work_end
        is_extended = is_workday and self.work_end <= hour < self.extended_end
        is_pre = is_workday and (self.work_start - self.precondition_hours) <= hour < self.work_start
        occupied = occupancy > 0.5 or is_working

        htg = s1_low + self.htg_margin
        clg = s1_high - self.clg_margin

        if clg - htg < self.min_deadband:
            mid = 0.5 * (s1_low + s1_high)
            htg = mid - self.min_deadband / 2.0
            clg = mid + self.min_deadband / 2.0

        if not occupied and not is_pre and not is_extended:
            htg = max(s2_low + 0.2, htg - self.night_htg_setback)
            clg = min(25.0, clg + self.night_clg_setup)

        if is_pre:
            htg += 0.3
            clg -= 0.3

        htg = max(18.0, min(25.0, htg))
        clg = max(18.0, min(25.0, clg))
        if htg > clg - 0.1:
            htg = clg - 0.1

        supply_air_temp = self._return_air_compensation(return_air_temp)

        co2_flow = self._co2_flow(co2_concentration)
        if is_pre:
            base_flow = self.flow_pre_flush
        elif occupied or is_extended:
            base_flow = self.flow_min_occupied
        elif not is_workday and self.work_start <= hour < self.work_end:
            base_flow = 0.5 * self.flow_min_occupied
        else:
            base_flow = self.flow_night

        flow = max(base_flow, co2_flow)
        flow = self._cold_weather_limit(flow, outdoor_temp)
        flow = max(0.0, min(1.0, flow))

        return htg, clg, supply_air_temp, flow
