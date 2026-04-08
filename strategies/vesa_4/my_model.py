"""Precision S1-Tracking Adaptive Controller with Thermal Mass Exploitation.

This strategy combines the best ideas from all hackathon approaches:
- Exact S1 comfort band computation from 24h rolling outdoor temperature
- Setpoints that hug the S1 band edges to minimize both energy and penalty
- Occupancy-aware CO2-predictive ventilation with hysteresis
- Thermal mass exploitation: let temperature drift within S1 to save energy
- Seasonal pre-conditioning and night setback
- Cold-weather outdoor air flow limiting
- Return-air compensated supply temperature

All tunable parameters are exposed as constructor arguments for Optuna optimization.
"""

from collections import deque
from typing import Tuple


class Vesa4Model:
    def __init__(
        self,
        # S1 band offsets: how far inside S1 to place setpoints (larger = safer but more energy)
        htg_margin: float = 0.3,
        clg_margin: float = 0.3,
        # Night setback offsets (how much to relax setpoints when unoccupied)
        night_htg_setback: float = 0.8,
        night_clg_setup: float = 0.8,
        # Pre-conditioning: how many hours before occupancy to start conditioning
        precondition_hours: float = 1.0,
        # Supply air temperature parameters
        sup_temp_cold: float = 19.0,
        sup_temp_warm: float = 17.5,
        return_air_cold_ref: float = 21.5,
        return_air_warm_ref: float = 24.5,
        # CO2 ventilation parameters
        co2_low_threshold: float = 600.0,
        co2_high_threshold: float = 750.0,
        co2_emergency: float = 900.0,
        # Fan flow parameters
        flow_min_occupied: float = 0.15,
        flow_moderate: float = 0.35,
        flow_boost: float = 1.0,
        flow_night: float = 0.0,
        flow_pre_flush: float = 0.40,
        # Cold weather limiting
        cold_limit_low: float = -25.0,
        cold_limit_high: float = -15.0,
        # Working hours
        work_start: float = 6.0,
        work_end: float = 18.0,
        extended_end: float = 22.0,
        # Minimum deadband between heating and cooling setpoints
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
        self.flow_moderate = flow_moderate
        self.flow_boost = flow_boost
        self.flow_night = flow_night
        self.flow_pre_flush = flow_pre_flush
        self.cold_limit_low = cold_limit_low
        self.cold_limit_high = cold_limit_high
        self.work_start = work_start
        self.work_end = work_end
        self.extended_end = extended_end
        self.min_deadband = min_deadband

        # 24h rolling outdoor temperature buffer (96 steps at 15-min interval = 24h)
        self._outdoor_history: deque = deque(maxlen=96)
        self._co2_rising: bool = False

    def _outdoor_24h_mean(self, outdoor_temp: float) -> float:
        self._outdoor_history.append(outdoor_temp)
        return sum(self._outdoor_history) / len(self._outdoor_history)

    def _s1_lower(self, t_mean: float) -> float:
        """S1 lower temperature limit from 24h rolling outdoor mean."""
        if t_mean <= 0.0:
            return 20.5
        elif t_mean <= 20.0:
            return 20.5 + 0.075 * t_mean
        else:
            return 22.0

    def _s1_upper(self, t_mean: float) -> float:
        """S1 upper temperature limit from 24h rolling outdoor mean."""
        if t_mean <= 0.0:
            return 22.0
        elif t_mean <= 15.0:
            return 22.0 + (25.0 - 22.0) / 15.0 * t_mean
        else:
            return 25.0

    def _s2_lower(self, t_mean: float) -> float:
        if t_mean <= 0.0:
            return 20.5
        elif t_mean <= 20.0:
            return 20.5 + 0.025 * t_mean
        else:
            return 21.0

    def _s2_upper(self, t_mean: float) -> float:
        if t_mean <= 0.0:
            return 23.0
        elif t_mean <= 15.0:
            return 23.0 + 0.20 * t_mean
        else:
            return 26.0

    def _return_air_compensation(self, return_air_temp: float) -> float:
        if return_air_temp <= self.return_air_cold_ref:
            return self.sup_temp_cold
        if return_air_temp >= self.return_air_warm_ref:
            return self.sup_temp_warm
        slope = (self.sup_temp_warm - self.sup_temp_cold) / (
            self.return_air_warm_ref - self.return_air_cold_ref
        )
        result = self.sup_temp_cold + slope * (return_air_temp - self.return_air_cold_ref)
        return max(min(self.sup_temp_cold, self.sup_temp_warm),
                   min(max(self.sup_temp_cold, self.sup_temp_warm), result))

    def _co2_demand_flow(self, co2: float) -> float:
        """Proportional CO2-demand-controlled flow with hysteresis."""
        if co2 >= self.co2_emergency:
            self._co2_rising = True
            return self.flow_boost
        if co2 <= self.co2_low_threshold:
            self._co2_rising = False
            return 0.0
        if co2 >= self.co2_high_threshold:
            self._co2_rising = True
            return self.flow_boost

        # Hysteresis: if CO2 was rising and hasn't dropped below low, keep moderate flow
        if self._co2_rising and co2 > self.co2_low_threshold + 30.0:
            fraction = (co2 - self.co2_low_threshold) / (
                self.co2_high_threshold - self.co2_low_threshold
            )
            return self.flow_min_occupied + fraction * (self.flow_boost - self.flow_min_occupied)

        fraction = (co2 - self.co2_low_threshold) / (
            self.co2_high_threshold - self.co2_low_threshold
        )
        return self.flow_min_occupied + fraction * (self.flow_boost - self.flow_min_occupied)

    def _cold_weather_limit(self, flow: float, outdoor_temp: float) -> float:
        """Limit ventilation flow during extreme cold to protect the system."""
        if outdoor_temp >= self.cold_limit_high:
            return flow
        if outdoor_temp <= self.cold_limit_low:
            return min(flow, self.flow_min_occupied)
        slope = (1.0 - self.flow_min_occupied / max(flow, 0.01)) / (
            self.cold_limit_high - self.cold_limit_low
        )
        max_fraction = self.flow_min_occupied / max(flow, 0.01) + slope * (
            outdoor_temp - self.cold_limit_low
        )
        return flow * min(1.0, max(self.flow_min_occupied / max(flow, 0.01), max_fraction))

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
        # 1. Compute 24h rolling outdoor mean and exact S1 comfort bands
        t_mean = self._outdoor_24h_mean(outdoor_temp)
        s1_lower = self._s1_lower(t_mean)
        s1_upper = self._s1_upper(t_mean)
        s2_lower = self._s2_lower(t_mean)

        # 2. Determine occupancy schedule
        # EnergyPlus day_of_week: 1=Sunday, 7=Saturday
        is_workday = day not in (1, 7)
        is_working = is_workday and self.work_start <= hour < self.work_end
        is_extended = is_workday and self.work_end <= hour < self.extended_end
        is_pre_condition = is_workday and (self.work_start - self.precondition_hours) <= hour < self.work_start

        occupied = occupancy > 0.5 or is_working

        # 3. Heating and cooling setpoints — precision S1 band tracking
        # Place heating setpoint just above S1 lower to avoid penalty
        # Place cooling setpoint just below S1 upper to avoid penalty
        htg = s1_lower + self.htg_margin
        clg = s1_upper - self.clg_margin

        # Enforce minimum deadband
        if clg - htg < self.min_deadband:
            mid = (s1_lower + s1_upper) / 2.0
            htg = mid - self.min_deadband / 2.0
            clg = mid + self.min_deadband / 2.0

        # Night setback: relax setpoints during unoccupied hours to save energy
        if not occupied and not is_pre_condition and not is_extended:
            htg_relaxed = htg - self.night_htg_setback
            clg_relaxed = clg + self.night_clg_setup
            # Don't go below S2 lower limit (avoids S2/S3 penalties)
            htg = max(s2_lower + 0.2, htg_relaxed)
            clg = min(25.0, clg_relaxed)

        # During pre-conditioning, tighten setpoints to hit target by work start
        if is_pre_condition:
            htg = s1_lower + self.htg_margin + 0.3
            clg = s1_upper - self.clg_margin - 0.3
            if clg - htg < self.min_deadband:
                mid = (s1_lower + s1_upper) / 2.0
                htg = mid - self.min_deadband / 2.0
                clg = mid + self.min_deadband / 2.0

        # Extended hours: slightly relaxed but still within S1
        if is_extended:
            htg = s1_lower + self.htg_margin * 0.5
            clg = s1_upper - self.clg_margin * 0.5
            if clg - htg < self.min_deadband:
                mid = (s1_lower + s1_upper) / 2.0
                htg = mid - self.min_deadband / 2.0
                clg = mid + self.min_deadband / 2.0

        # Clamp to physical actuator limits
        htg = max(18.0, min(25.0, htg))
        clg = max(18.0, min(25.0, clg))
        if htg > clg:
            htg = clg - 0.1

        # 4. Supply air temperature
        supply_air_temp = self._return_air_compensation(return_air_temp)

        # 5. Fan flow rate
        co2_flow = self._co2_demand_flow(co2_concentration)

        # Schedule-based base flow
        if is_pre_condition:
            base_flow = self.flow_pre_flush
        elif occupied:
            base_flow = self.flow_min_occupied
        elif is_extended:
            base_flow = self.flow_min_occupied
        elif not is_workday and self.work_start <= hour < self.work_end:
            base_flow = self.flow_min_occupied * 0.5
        else:
            base_flow = self.flow_night

        flow = max(co2_flow, base_flow)

        # Cold weather flow limiting
        flow = self._cold_weather_limit(flow, outdoor_temp)

        # Clamp to actuator limits
        flow = max(0.0, min(1.0, flow))

        return htg, clg, supply_air_temp, flow
