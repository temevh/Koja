"""Vesa6 hybrid rule-based HVAC controller.

Strategy combines three tactics:
1) S1 comfort-band tracking from a rolling outdoor average,
2) Return-air-compensated supply-air temperature,
3) CO2 + schedule + occupancy driven ventilation with free-cooling night flush.
"""

from collections import deque
from typing import Tuple


class Vesa6Model:
    def __init__(
        self,
        htg_margin: float = 0.35,
        clg_margin: float = 0.30,
        night_htg_setback: float = 0.25,
        night_clg_setup: float = 0.25,
        preflush_start: float = 5.5,
        work_start: float = 7.0,
        work_end: float = 18.0,
        evening_end: float = 21.0,
        sup_temp_cold: float = 20.5,
        sup_temp_warm: float = 17.0,
        return_air_cold_ref: float = 21.0,
        return_air_warm_ref: float = 25.0,
        flow_work: float = 0.22,
        flow_evening: float = 0.10,
        flow_night: float = 0.02,
        flow_preflush: float = 0.28,
        flow_co2_boost: float = 1.0,
        co2_low: float = 610.0,
        co2_high: float = 770.0,
        co2_emergency: float = 930.0,
        cold_limit_low: float = -25.0,
        cold_limit_high: float = -12.0,
        nightflush_delta: float = 1.5,
        nightflush_min_temp: float = 22.0,
        nightflush_flow: float = 0.35,
        min_deadband: float = 1.0,
        supply_override_gap: float = 0.5,
        temp_recovery_gap: float = 0.40,
        temp_recovery_flow: float = 0.55,
        s2_recovery_flow: float = 0.85,
        s2_recovery_margin: float = 0.0,
    ) -> None:
        self.htg_margin = htg_margin
        self.clg_margin = clg_margin
        self.night_htg_setback = night_htg_setback
        self.night_clg_setup = night_clg_setup
        self.preflush_start = preflush_start
        self.work_start = work_start
        self.work_end = work_end
        self.evening_end = evening_end
        self.sup_temp_cold = sup_temp_cold
        self.sup_temp_warm = sup_temp_warm
        self.return_air_cold_ref = return_air_cold_ref
        self.return_air_warm_ref = return_air_warm_ref
        self.flow_work = flow_work
        self.flow_evening = flow_evening
        self.flow_night = flow_night
        self.flow_preflush = flow_preflush
        self.flow_co2_boost = flow_co2_boost
        self.co2_low = co2_low
        self.co2_high = co2_high
        self.co2_emergency = co2_emergency
        self.cold_limit_low = cold_limit_low
        self.cold_limit_high = cold_limit_high
        self.nightflush_delta = nightflush_delta
        self.nightflush_min_temp = nightflush_min_temp
        self.nightflush_flow = nightflush_flow
        self.min_deadband = min_deadband
        self.supply_override_gap = supply_override_gap
        self.temp_recovery_gap = temp_recovery_gap
        self.temp_recovery_flow = temp_recovery_flow
        self.s2_recovery_flow = s2_recovery_flow
        self.s2_recovery_margin = s2_recovery_margin

        self._outdoor_history = deque(maxlen=96)  # 24 h @ 15 min timestep

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
        """Calculate 4 actuator setpoints for one simulation step."""
        t_out_24h = self._update_outdoor_24h(outdoor_temp)
        s1_low, s1_high = self._s1_band(t_out_24h)
        s2_low, s2_high = self._s2_band(t_out_24h)

        is_workday = day not in (1, 7)
        is_preflush = is_workday and self.preflush_start <= hour < self.work_start
        is_work = is_workday and self.work_start <= hour < self.work_end
        is_evening = is_workday and self.work_end <= hour < self.evening_end
        occupied_now = occupancy > 0.5 or is_work

        htg = s1_low + self.htg_margin
        clg = s1_high - self.clg_margin

        if not occupied_now and not is_preflush:
            htg = max(s2_low + 0.2, htg - self.night_htg_setback)
            clg = min(25.0, clg + self.night_clg_setup)

        if is_preflush:
            htg += 0.25
            clg -= 0.20

        htg, clg = self._clamp_zone_setpoints(htg, clg)
        supply_air_temp = self._return_air_compensation(return_air_temp)
        if zone_temp < htg - self.supply_override_gap:
            supply_air_temp = 21.0
        elif zone_temp > clg + self.supply_override_gap:
            supply_air_temp = 16.0

        base_flow = self._scheduled_base_flow(
            is_work=is_work,
            is_evening=is_evening,
            is_preflush=is_preflush,
            occupied_now=occupied_now,
        )
        co2_flow = self._co2_flow(co2_concentration)
        fan_flow_rate = max(base_flow, co2_flow)
        fan_flow_rate = self._maybe_apply_night_flush(
            flow=fan_flow_rate,
            outdoor_temp=outdoor_temp,
            zone_temp=zone_temp,
            occupied_now=occupied_now,
            is_work=is_work,
        )
        needs_temp_recovery = (
            zone_temp < (htg - self.temp_recovery_gap)
            or zone_temp > (clg + self.temp_recovery_gap)
        )
        outside_s2 = (
            zone_temp < (s2_low - self.s2_recovery_margin)
            or zone_temp > (s2_high + self.s2_recovery_margin)
        )
        if outside_s2:
            fan_flow_rate = max(fan_flow_rate, self.s2_recovery_flow)
            if zone_temp < s2_low:
                supply_air_temp = 21.0
            elif zone_temp > s2_high:
                supply_air_temp = 16.0
        elif needs_temp_recovery and (occupied_now or is_preflush):
            fan_flow_rate = max(fan_flow_rate, self.temp_recovery_flow)
        fan_flow_rate = self._cold_weather_limit(fan_flow_rate, outdoor_temp)
        fan_flow_rate = max(0.0, min(1.0, fan_flow_rate))

        return htg, clg, supply_air_temp, fan_flow_rate

    def _update_outdoor_24h(self, outdoor_temp: float) -> float:
        self._outdoor_history.append(float(outdoor_temp))
        return sum(self._outdoor_history) / len(self._outdoor_history)

    @staticmethod
    def _s1_band(t_mean: float) -> Tuple[float, float]:
        if t_mean <= 0.0:
            lower = 20.5
            upper = 22.0
        elif t_mean <= 20.0:
            lower = 20.5 + 0.075 * t_mean
            if t_mean <= 15.0:
                upper = 22.0 + 0.2 * t_mean
            else:
                upper = 25.0
        else:
            lower = 22.0
            upper = 25.0
        return lower, upper

    @staticmethod
    def _s2_band(t_mean: float) -> Tuple[float, float]:
        if t_mean <= 0.0:
            lower = 20.5
            upper = 23.0
            return lower, upper
        if t_mean <= 20.0:
            lower = 20.5 + 0.025 * t_mean
        else:
            lower = 21.0
        if t_mean <= 15.0:
            upper = 23.0 + 0.2 * t_mean
        else:
            upper = 26.0
        return lower, upper

    def _return_air_compensation(self, return_air_temp: float) -> float:
        if return_air_temp <= self.return_air_cold_ref:
            return self.sup_temp_cold
        if return_air_temp >= self.return_air_warm_ref:
            return self.sup_temp_warm
        ratio = (return_air_temp - self.return_air_cold_ref) / max(
            self.return_air_warm_ref - self.return_air_cold_ref, 0.001
        )
        supply = self.sup_temp_cold + ratio * (self.sup_temp_warm - self.sup_temp_cold)
        return max(16.0, min(21.0, supply))

    def _scheduled_base_flow(
        self, is_work: bool, is_evening: bool, is_preflush: bool, occupied_now: bool
    ) -> float:
        if is_preflush:
            return self.flow_preflush
        if is_work:
            return self.flow_work
        if is_evening:
            return self.flow_evening
        if occupied_now:
            return max(self.flow_evening, 0.5 * self.flow_work)
        return self.flow_night

    def _co2_flow(self, co2_concentration: float) -> float:
        if co2_concentration >= self.co2_emergency:
            return self.flow_co2_boost
        if co2_concentration <= self.co2_low:
            return 0.0
        if co2_concentration >= self.co2_high:
            return self.flow_co2_boost
        ratio = (co2_concentration - self.co2_low) / max(self.co2_high - self.co2_low, 1.0)
        return self.flow_work + ratio * (self.flow_co2_boost - self.flow_work)

    def _maybe_apply_night_flush(
        self,
        flow: float,
        outdoor_temp: float,
        zone_temp: float,
        occupied_now: bool,
        is_work: bool,
    ) -> float:
        should_flush = (
            not occupied_now
            and not is_work
            and zone_temp >= self.nightflush_min_temp
            and outdoor_temp < zone_temp - self.nightflush_delta
        )
        if should_flush:
            return max(flow, self.nightflush_flow)
        return flow

    def _cold_weather_limit(self, flow: float, outdoor_temp: float) -> float:
        if outdoor_temp >= self.cold_limit_high:
            return flow
        if outdoor_temp <= self.cold_limit_low:
            return min(flow, self.flow_work)
        ratio = (outdoor_temp - self.cold_limit_low) / max(
            self.cold_limit_high - self.cold_limit_low, 0.001
        )
        allowed_max = self.flow_work + ratio * (1.0 - self.flow_work)
        return min(flow, allowed_max)

    def _clamp_zone_setpoints(self, htg: float, clg: float) -> Tuple[float, float]:
        htg = max(18.0, min(25.0, htg))
        clg = max(18.0, min(25.0, clg))
        if clg - htg < self.min_deadband:
            mid = 0.5 * (htg + clg)
            htg = mid - self.min_deadband / 2.0
            clg = mid + self.min_deadband / 2.0
            htg = max(18.0, min(25.0, htg))
            clg = max(18.0, min(25.0, clg))
        if htg > clg:
            htg = clg
        return htg, clg
