"""Parameterized RBC model for Bayesian Optimization.

Every constant from the hand-tuned RBC is replaced by a parameter
that Optuna can search over. The control logic structure is identical
to rbc_scheduled/rbc_model.py.
"""

from typing import Tuple

import numpy as np


# Default parameters — matches rbc_scheduled baseline
DEFAULT_PARAMS: dict = {
    # Zone setpoint compensation
    "htg_setpoint_low": 21.0,
    "htg_setpoint_high": 22.0,
    "clg_setpoint_low": 23.0,
    "clg_setpoint_high": 25.0,
    "outdoor_temp_low": 0.0,
    "outdoor_temp_high": 20.0,
    # Supply air temperature compensation
    "return_air_temp_low": 21.5,
    "return_air_temp_high": 24.5,
    "supply_temp_at_low": 19.0,
    "supply_temp_at_high": 17.0,
    # CO2 demand-controlled ventilation
    "co2_min_limit": 500.0,
    "co2_max_limit": 750.0,
    # Fan flow rates
    "flow_off": 0.0,
    "flow_low": 0.20,
    "flow_moderate": 0.60,
    "flow_boost": 1.0,
    # Schedule timing (workday)
    "nightflush_start": 1.5,
    "nightflush_duration": 1.0,
    "prework_start": 4.5,
    "prework_duration": 1.0,
    "work_start": 5.5,
    "work_end": 23.5,
    # Outdoor temperature limits for flow
    "outdoor_temp_low_limit": -25.0,
    "outdoor_temp_high_limit": -15.0,
}


class BOModel:
    """Parameterized HVAC controller whose constants come from a dict."""

    def __init__(self, params: dict | None = None) -> None:
        self.p = {**DEFAULT_PARAMS, **(params or {})}

    # -- Zone setpoints (outdoor-temperature-compensated) ---------------------

    def _zone_setpoints(self, outdoor_temp: float) -> Tuple[float, float]:
        t = np.clip(
            (outdoor_temp - self.p["outdoor_temp_low"])
            / (self.p["outdoor_temp_high"] - self.p["outdoor_temp_low"]),
            0.0,
            1.0,
        )
        htg = self.p["htg_setpoint_low"] + t * (
            self.p["htg_setpoint_high"] - self.p["htg_setpoint_low"]
        )
        clg = self.p["clg_setpoint_low"] + t * (
            self.p["clg_setpoint_high"] - self.p["clg_setpoint_low"]
        )
        # Enforce htg <= clg
        if htg > clg:
            mid = (htg + clg) / 2.0
            htg, clg = mid, mid
        return float(htg), float(clg)

    # -- Supply air temperature (return-air-compensated) ----------------------

    def _supply_temp(self, return_air_temp: float) -> float:
        lo = self.p["return_air_temp_low"]
        hi = self.p["return_air_temp_high"]
        sup_lo = self.p["supply_temp_at_low"]
        sup_hi = self.p["supply_temp_at_high"]

        if return_air_temp <= lo:
            return sup_lo
        if return_air_temp >= hi:
            return sup_hi

        slope = (sup_hi - sup_lo) / (hi - lo)
        val = sup_lo + slope * (return_air_temp - lo)
        return float(np.clip(val, min(sup_lo, sup_hi), max(sup_lo, sup_hi)))

    # -- Fan flow (schedule + CO2 boost + outdoor limit) ----------------------

    def _fan_flow(
        self,
        hour: float,
        day: int,
        co2_concentration: float,
        outdoor_temp: float,
    ) -> float:
        p = self.p
        is_workday = day != 1 and day != 7

        nf_start = p["nightflush_start"]
        nf_end = nf_start + p["nightflush_duration"]
        pw_start = p["prework_start"]
        pw_end = pw_start + p["prework_duration"]
        w_start = p["work_start"]
        w_end = p["work_end"]

        # Schedule-based base flow
        if is_workday:
            if pw_start <= hour < pw_end:
                base_flow = p["flow_moderate"]
            elif nf_start <= hour < nf_end:
                base_flow = p["flow_low"]
            elif w_start <= hour < w_end:
                base_flow = p["flow_low"]
            else:
                base_flow = p["flow_off"]
        else:
            # Weekend: low during 0-2h and 6-17h, off otherwise
            if 0.0 <= hour < 2.0 or 6.0 <= hour < 17.0:
                base_flow = p["flow_low"]
            else:
                base_flow = p["flow_off"]

        # CO2 demand boost
        co2_min = p["co2_min_limit"]
        co2_max = p["co2_max_limit"]
        if co2_concentration <= co2_min:
            co2_flow = 0.0
        elif co2_concentration >= co2_max:
            co2_flow = p["flow_boost"]
        else:
            frac = (co2_concentration - co2_min) / (co2_max - co2_min)
            co2_flow = p["flow_low"] + frac * (p["flow_boost"] - p["flow_low"])

        target_flow = max(co2_flow, base_flow)

        # Outdoor temperature limit (cold protection)
        ot_lo = p["outdoor_temp_low_limit"]
        ot_hi = p["outdoor_temp_high_limit"]
        if outdoor_temp <= ot_lo:
            max_allowed = p["flow_low"]
        elif outdoor_temp >= ot_hi:
            max_allowed = p["flow_boost"]
        else:
            slope = (p["flow_boost"] - p["flow_low"]) / (ot_hi - ot_lo)
            max_allowed = p["flow_low"] + slope * (outdoor_temp - ot_lo)

        return min(target_flow, max_allowed)

    # -- Main interface -------------------------------------------------------

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
        htg, clg = self._zone_setpoints(outdoor_temp)
        supply = self._supply_temp(return_air_temp)
        flow = self._fan_flow(hour, day, co2_concentration, outdoor_temp)
        return htg, clg, supply, flow
