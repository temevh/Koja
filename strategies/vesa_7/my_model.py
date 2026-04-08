"""Vesa7 two-level HVAC controller with forecast preconditioning."""

from __future__ import annotations

import csv
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple


class Vesa7Model:
    """Supervisor + fast local policy controller.

    - Supervisor mode updates every 1-2 hours:
      comfort / economy / purge / recovery
    - Low-level policy runs each timestep and maps mode -> setpoints/flow.
    - Uses EPW dry-bulb forecast and simple thermal trend estimation for
      preconditioning decisions.
    - Tracks "budget pressure" for comfort/IAQ drift (90% style allowance)
      and tightens control when budget burn is too fast.
    - Adds robust guardbands from forecast spread + recent disturbance.
    """

    def __init__(
        self,
        epw_file: Optional[Path] = None,
        work_start: float = 7.0,
        work_end: float = 18.0,
        min_deadband: float = 1.0,
        co2_low: float = 600.0,
        co2_high: float = 780.0,
        co2_emergency: float = 950.0,
        sup_temp_cold: float = 20.0,
        sup_temp_warm: float = 17.0,
        return_air_cold_ref: float = 21.0,
        return_air_warm_ref: float = 25.0,
        cold_limit_low: float = -25.0,
        cold_limit_high: float = -12.0,
        budget_fraction: float = 0.10,
        supervisor_interval_occ_steps: int = 4,
        supervisor_interval_unocc_steps: int = 8,
        comfort_htg_offset: float = 0.25,
        comfort_clg_offset: float = 0.20,
        comfort_flow_occ: float = 0.22,
        comfort_flow_unocc: float = 0.16,
        recovery_flow: float = 0.42,
        purge_flow: float = 0.55,
        economy_flow: float = 0.04,
        co2_flow_min: float = 0.18,
        co2_flow_high: float = 0.75,
        forecast_horizon_h: int = 6,
        forecast_cold_risk: float = -2.0,
        forecast_hot_risk: float = 24.0,
    ) -> None:
        self.current_mode = "economy"
        self._step_idx = 0
        self._last_supervisor_step = -999

        self._outdoor_roll = deque(maxlen=96)    # 24 h @ 15 min steps
        self._zone_temp_roll = deque(maxlen=16)  # 4 h trend window
        self._zone_delta_roll = deque(maxlen=16)
        self._last_zone_temp: Optional[float] = None

        self._epw_dry_bulb = self._load_epw_dry_bulb(epw_file)

        # Core tunables
        self.work_start = float(work_start)
        self.work_end = float(work_end)
        self.min_deadband = float(min_deadband)

        self.co2_low = float(co2_low)
        self.co2_high = float(co2_high)
        self.co2_emergency = float(co2_emergency)

        self.sup_temp_cold = float(sup_temp_cold)
        self.sup_temp_warm = float(sup_temp_warm)
        self.return_air_cold_ref = float(return_air_cold_ref)
        self.return_air_warm_ref = float(return_air_warm_ref)

        self.cold_limit_low = float(cold_limit_low)
        self.cold_limit_high = float(cold_limit_high)

        # Budget-aware control:
        # Keep severe violations below roughly 10% of elapsed steps.
        self.budget_fraction = float(budget_fraction)
        self._temp_violation_steps = 0.0
        self._co2_violation_steps = 0.0

        # Supervisor tuning
        self.supervisor_interval_occ_steps = max(1, int(supervisor_interval_occ_steps))
        self.supervisor_interval_unocc_steps = max(1, int(supervisor_interval_unocc_steps))

        # Local policy tuning
        self.comfort_htg_offset = float(comfort_htg_offset)
        self.comfort_clg_offset = float(comfort_clg_offset)
        self.comfort_flow_occ = float(comfort_flow_occ)
        self.comfort_flow_unocc = float(comfort_flow_unocc)
        self.recovery_flow = float(recovery_flow)
        self.purge_flow = float(purge_flow)
        self.economy_flow = float(economy_flow)
        self.co2_flow_min = float(co2_flow_min)
        self.co2_flow_high = float(co2_flow_high)

        # Forecast tuning
        self.forecast_horizon_h = max(2, int(forecast_horizon_h))
        self.forecast_cold_risk = float(forecast_cold_risk)
        self.forecast_hot_risk = float(forecast_hot_risk)

    def calculate_setpoints(
        self,
        zone_temp: float,
        outdoor_temp: float,
        return_air_temp: float,
        occupancy: float,
        hour: float,
        day: int,
        co2_concentration: float,
        month: int,
        day_of_month: int,
    ) -> Tuple[float, float, float, float]:
        self._step_idx += 1
        self._outdoor_roll.append(float(outdoor_temp))
        self._zone_temp_roll.append(float(zone_temp))
        if self._last_zone_temp is not None:
            self._zone_delta_roll.append(abs(zone_temp - self._last_zone_temp))
        self._last_zone_temp = zone_temp

        t24 = sum(self._outdoor_roll) / len(self._outdoor_roll)
        s1_low, s1_high = self._s1_band(t24)
        s2_low, _s2_high = self._s2_band(t24)

        self._update_budget_usage(zone_temp, co2_concentration, s1_low, s1_high)
        budget_pressure = self._budget_pressure()

        occupied_sched = self._is_scheduled_occupied(day, hour)
        occupied_now = occupancy > 0.5 or occupied_sched

        if self._should_update_supervisor(occupied_now):
            self.current_mode = self._supervisor_mode(
                zone_temp=zone_temp,
                outdoor_temp=outdoor_temp,
                co2=co2_concentration,
                occupied_now=occupied_now,
                day=day,
                hour=hour,
                month=month,
                day_of_month=day_of_month,
                s1_low=s1_low,
                s1_high=s1_high,
                budget_pressure=budget_pressure,
            )
            self._last_supervisor_step = self._step_idx

        f_min, f_max = self._forecast_min_max(
            month, day_of_month, hour, horizon_h=self.forecast_horizon_h
        )
        forecast_spread = 0.0
        if f_min is not None and f_max is not None:
            forecast_spread = max(0.0, f_max - f_min)

        htg, clg, supply, flow = self._local_policy(
            mode=self.current_mode,
            zone_temp=zone_temp,
            outdoor_temp=outdoor_temp,
            return_air_temp=return_air_temp,
            co2=co2_concentration,
            occupied_now=occupied_now,
            s1_low=s1_low,
            s1_high=s1_high,
            s2_low=s2_low,
            budget_pressure=budget_pressure,
            forecast_spread=forecast_spread,
        )

        htg, clg = self._clamp_zone_setpoints(htg, clg)
        supply = max(16.0, min(21.0, supply))
        flow = self._cold_weather_limit(max(0.0, min(1.0, flow)), outdoor_temp)
        return htg, clg, supply, flow

    def _should_update_supervisor(self, occupied_now: bool) -> bool:
        interval_steps = (
            self.supervisor_interval_occ_steps
            if occupied_now
            else self.supervisor_interval_unocc_steps
        )
        return (self._step_idx - self._last_supervisor_step) >= interval_steps

    def _supervisor_mode(
        self,
        zone_temp: float,
        outdoor_temp: float,
        co2: float,
        occupied_now: bool,
        day: int,
        hour: float,
        month: int,
        day_of_month: int,
        s1_low: float,
        s1_high: float,
        budget_pressure: float,
    ) -> str:
        if co2 >= self.co2_emergency:
            return "recovery"

        if budget_pressure > 1.35:
            return "recovery"

        if occupied_now:
            if zone_temp < s1_low - 0.4 or zone_temp > s1_high + 0.4:
                return "recovery"
            return "comfort"

        if self._needs_forecast_preconditioning(
            zone_temp=zone_temp,
            day=day,
            hour=hour,
            month=month,
            day_of_month=day_of_month,
            s1_low=s1_low,
            s1_high=s1_high,
        ):
            return "recovery"

        purge_ok = (
            zone_temp > max(22.2, s1_high)
            and outdoor_temp < zone_temp - 1.5
            and co2 < self.co2_high
        )
        if purge_ok:
            return "purge"

        return "economy"

    def _local_policy(
        self,
        mode: str,
        zone_temp: float,
        outdoor_temp: float,
        return_air_temp: float,
        co2: float,
        occupied_now: bool,
        s1_low: float,
        s1_high: float,
        s2_low: float,
        budget_pressure: float,
        forecast_spread: float,
    ) -> Tuple[float, float, float, float]:
        disturbance = self._disturbance_level()
        robust_margin = self._robust_margin(forecast_spread, disturbance, budget_pressure)
        co2_flow = self._co2_flow(co2, budget_pressure)
        supply = self._return_air_compensation(return_air_temp)

        if mode == "comfort":
            htg = s1_low + self.comfort_htg_offset + robust_margin
            clg = s1_high - self.comfort_clg_offset - robust_margin
            base_flow = self.comfort_flow_occ if occupied_now else self.comfort_flow_unocc
        elif mode == "recovery":
            mid = 0.5 * (s1_low + s1_high)
            htg = mid - 0.35
            clg = mid + 0.35
            base_flow = self.recovery_flow
            if zone_temp > clg + 0.4:
                supply = 16.2
            elif zone_temp < htg - 0.4:
                supply = 21.0
        elif mode == "purge":
            htg = max(18.0, s2_low + 0.2)
            clg = min(25.0, s1_high + 0.8)
            base_flow = self.purge_flow
            supply = 16.0
        else:  # economy
            # Budget manager: when pressure low, allow wider drift; when high, tighten.
            economy_relax = self._economy_relaxation(budget_pressure)
            htg = max(18.0, s2_low + 0.15 - economy_relax)
            clg = min(25.0, s1_high + 0.8 + economy_relax)
            base_flow = self.economy_flow

        flow = max(base_flow, co2_flow)
        return htg, clg, supply, flow

    def _needs_forecast_preconditioning(
        self,
        zone_temp: float,
        day: int,
        hour: float,
        month: int,
        day_of_month: int,
        s1_low: float,
        s1_high: float,
    ) -> bool:
        hours_to_occ = self._hours_to_next_occupancy(day, hour)
        if hours_to_occ is None or hours_to_occ > 6.0:
            return False

        slope = self._zone_temp_slope_per_hour()
        pred_temp = zone_temp + slope * hours_to_occ
        comfort_mid = 0.5 * (s1_low + s1_high)

        f_min, f_max = self._forecast_min_max(
            month,
            day_of_month,
            hour,
            horizon_h=max(2, int(hours_to_occ) + 1),
        )
        if f_min is None or f_max is None:
            return hours_to_occ <= 2.0 and abs(pred_temp - comfort_mid) > 0.8

        cold_risk = f_min <= self.forecast_cold_risk
        hot_risk = f_max >= self.forecast_hot_risk
        if cold_risk and pred_temp < comfort_mid - 0.35:
            return True
        if hot_risk and pred_temp > comfort_mid + 0.35:
            return True
        if abs(pred_temp - comfort_mid) > 1.0 and hours_to_occ <= 2.5:
            return True
        return False

    def _zone_temp_slope_per_hour(self) -> float:
        if len(self._zone_temp_roll) < 4:
            return 0.0
        first = self._zone_temp_roll[0]
        last = self._zone_temp_roll[-1]
        elapsed_h = (len(self._zone_temp_roll) - 1) * 0.25
        return (last - first) / max(elapsed_h, 0.25)

    def _co2_flow(self, co2: float, budget_pressure: float) -> float:
        low = self.co2_low
        high = self.co2_high
        if budget_pressure > 1.0:
            # Tighten IAQ response if we are burning budget too fast.
            low -= 40.0
            high -= 40.0

        if co2 <= low:
            return 0.0
        if co2 >= self.co2_emergency:
            return 1.0
        if co2 >= high:
            return self.co2_flow_high
        ratio = (co2 - low) / max(high - low, 1.0)
        return self.co2_flow_min + ratio * (self.co2_flow_high - self.co2_flow_min)

    def _update_budget_usage(
        self, zone_temp: float, co2: float, s1_low: float, s1_high: float
    ) -> None:
        # Use severe conditions as budget spends.
        if zone_temp < (s1_low - 0.35) or zone_temp > (s1_high + 0.35):
            self._temp_violation_steps += 1.0
        if co2 > 770.0:
            self._co2_violation_steps += 1.0

    def _budget_pressure(self) -> float:
        allowed = max(1.0, self.budget_fraction * float(self._step_idx))
        temp_p = self._temp_violation_steps / allowed
        co2_p = self._co2_violation_steps / allowed
        return max(temp_p, co2_p)

    def _disturbance_level(self) -> float:
        if not self._zone_delta_roll:
            return 0.0
        avg_abs_delta = sum(self._zone_delta_roll) / len(self._zone_delta_roll)
        # 0.2 C step-to-step is calm, >=0.8 C is very disturbed.
        return max(0.0, min(1.0, (avg_abs_delta - 0.2) / 0.6))

    @staticmethod
    def _robust_margin(forecast_spread: float, disturbance: float, budget_pressure: float) -> float:
        # Forecast spread and disturbance increase margin; low budget pressure can relax it.
        spread_term = max(0.0, min(1.0, (forecast_spread - 6.0) / 10.0))
        pressure_term = max(0.0, min(1.0, budget_pressure - 1.0))
        margin = 0.10 * spread_term + 0.15 * disturbance + 0.20 * pressure_term
        return max(0.0, min(0.35, margin))

    @staticmethod
    def _economy_relaxation(budget_pressure: float) -> float:
        # If budget usage is low, spend more comfort budget for energy savings.
        # If budget is tight, reduce relaxation.
        if budget_pressure < 0.6:
            return 0.45
        if budget_pressure < 0.9:
            return 0.25
        if budget_pressure < 1.1:
            return 0.10
        return 0.0

    def _return_air_compensation(self, return_air_temp: float) -> float:
        if return_air_temp <= self.return_air_cold_ref:
            return self.sup_temp_cold
        if return_air_temp >= self.return_air_warm_ref:
            return self.sup_temp_warm
        ratio = (return_air_temp - self.return_air_cold_ref) / max(
            self.return_air_warm_ref - self.return_air_cold_ref, 0.001
        )
        return self.sup_temp_cold + ratio * (self.sup_temp_warm - self.sup_temp_cold)

    def _cold_weather_limit(self, flow: float, outdoor_temp: float) -> float:
        if outdoor_temp >= self.cold_limit_high:
            return flow
        if outdoor_temp <= self.cold_limit_low:
            return min(flow, 0.22)
        ratio = (outdoor_temp - self.cold_limit_low) / max(
            self.cold_limit_high - self.cold_limit_low, 0.001
        )
        allowed_max = 0.22 + ratio * (1.0 - 0.22)
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

    @staticmethod
    def _s1_band(t_mean: float) -> Tuple[float, float]:
        if t_mean <= 0.0:
            lower = 20.5
            upper = 22.0
        elif t_mean <= 20.0:
            lower = 20.5 + 0.075 * t_mean
            upper = 22.0 + 0.2 * min(t_mean, 15.0)
        else:
            lower = 22.0
            upper = 25.0
        upper = min(25.0, upper)
        return lower, upper

    @staticmethod
    def _s2_band(t_mean: float) -> Tuple[float, float]:
        if t_mean <= 0.0:
            lower = 20.5
            upper = 23.0
        elif t_mean <= 20.0:
            lower = 20.5 + 0.025 * t_mean
            upper = 23.0 + 0.2 * min(t_mean, 15.0)
        else:
            lower = 21.0
            upper = 26.0
        return lower, upper

    def _is_scheduled_occupied(self, day: int, hour: float) -> bool:
        is_workday = day not in (1, 7)
        return is_workday and self.work_start <= hour < self.work_end

    def _hours_to_next_occupancy(self, day: int, hour: float) -> Optional[float]:
        h = float(hour)
        d = int(day)
        for offset_h in range(1, 72):
            h_check = h + offset_h
            day_add = int(h_check // 24)
            hour_norm = h_check % 24
            day_norm = ((d - 1 + day_add) % 7) + 1
            if self._is_scheduled_occupied(day_norm, hour_norm):
                return float(offset_h)
        return None

    def _forecast_min_max(
        self,
        month: int,
        day_of_month: int,
        hour: float,
        horizon_h: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        if not self._epw_dry_bulb:
            return None, None

        temps = []
        base_hour = int(hour)
        try:
            base = datetime(2024, int(month), int(day_of_month), base_hour, 0, 0)
        except ValueError:
            return None, None

        for dh in range(1, horizon_h + 1):
            t = base + timedelta(hours=dh)
            key = (t.month, t.day, t.hour)
            if key in self._epw_dry_bulb:
                temps.append(self._epw_dry_bulb[key])
        if not temps:
            return None, None
        return min(temps), max(temps)

    @staticmethod
    def _load_epw_dry_bulb(epw_file: Optional[Path]) -> Dict[Tuple[int, int, int], float]:
        if epw_file is None:
            return {}
        path = Path(epw_file)
        if not path.exists():
            return {}

        data: Dict[Tuple[int, int, int], float] = {}
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if idx < 8:
                        continue
                    if len(row) < 7:
                        continue
                    month = int(float(row[1]))
                    day = int(float(row[2]))
                    hour_1_to_24 = int(float(row[3]))
                    hour_0_to_23 = (hour_1_to_24 - 1) % 24
                    dry_bulb = float(row[6])
                    data[(month, day, hour_0_to_23)] = dry_bulb
        except Exception:
            return {}
        return data
