"""Rule-Based Control (RBC) model for AHU and zone setpoint management.

This module implements a rule-based controller that determines:
- Zone heating and cooling temperature setpoints (outdoor-compensated)
- Supply air temperature setpoint (return-air-compensated)
- AHU fan mass flow rate (schedule-based with CO2 demand boosting)

The control logic follows typical HVAC best practices for a DOAS
(Dedicated Outdoor Air System) with fan coil units.
"""

from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# Constants — tune these to optimise total cost
# ============================================================================

# Zone heating setpoint [°C] — linear ramp between outdoor temp bounds
# S1 lower limits: 20.5°C (outdoor ≤ 0°C) → 22.0°C (outdoor ≥ 20°C)
ZONE_HTG_SETPOINT_LOW  = 21.0   # heating setpoint when outdoor <= 0 °C  (0.5°C above S1 lower)
ZONE_HTG_SETPOINT_HIGH = 22.5   # heating setpoint when outdoor >= 20 °C (0.5°C above S1 lower)

# Zone cooling setpoint [°C] — linear ramp between outdoor temp bounds
# S1 upper limits: 22.0°C (outdoor ≤ 0°C) → 25.0°C (outdoor ≥ 15°C)
ZONE_CLG_SETPOINT_LOW  = 21.5   # cooling setpoint when outdoor <= 0 °C  (0.5°C below S1 upper)
ZONE_CLG_SETPOINT_HIGH = 24.5   # cooling setpoint when outdoor >= 20 °C (0.5°C below S1 upper)

# Outdoor temperature range for setpoint compensation [°C]
OUTDOOR_TEMP_LOW  = 0.0
OUTDOOR_TEMP_HIGH = 20.0

# Supply air temperature compensation curve (return-air-based)
# Near-neutral supply avoids fighting FCU heaters in winter (huge gas savings)
RETURN_AIR_TEMP_LOW  = 21.5   # return air temp at which supply = SUP_TEMP_AT_LOW
RETURN_AIR_TEMP_HIGH = 24.5   # return air temp at which supply = SUP_TEMP_AT_HIGH
SUP_TEMP_AT_LOW  = 20.0       # supply air temp when return air is cool (was 19)
SUP_TEMP_AT_HIGH = 19.0       # supply air temp when return air is warm (was 18)

# CO2 demand-controlled ventilation thresholds [ppm]
CO2_MIN_LIMIT = 500   # below this → minimum ventilation is enough
CO2_MAX_LIMIT = 750   # above this → full boost

# Extreme cold outdoor temperature limits for fan flow capping [°C]
OUTDOOR_TEMP_COLD_LIMIT = -25.0   # at or below → cap flow to FLOW_LOW
OUTDOOR_TEMP_COLD_UPPER = -15.0   # at or above → no cap

# AHU fan mass flow rates [kg/s]
FLOW_LOW      = 0.20   # minimum / weekend / night
FLOW_MODERATE = 0.40   # pre-flush (was 0.60 — reduced to save energy)
FLOW_BOOST    = 1.0    # CO2 boost during occupied hours
FLOW_MAX      = 1.0    # physical upper limit


@dataclass
class AHUModeParams:
    """Parameters for a single AHU operating mode."""
    flow: float
    mode_name: str


class RBCModel:
    """Rule-based HVAC controller for a 5-zone DOAS building.

    The model computes heating/cooling setpoints, supply air temperature,
    and fan flow rate at every simulation timestep.  It is intended to be
    called from an EnergyPlus runtime callback.
    """

    def __init__(self) -> None:
        # Current zone setpoints (updated each timestep)
        self.htg_setpoint: float = ZONE_HTG_SETPOINT_LOW
        self.clg_setpoint: float = ZONE_CLG_SETPOINT_LOW

        # CO2 demand-boosting state
        self._co2_boosting: bool = False

        # Named operating modes (for logging / reference)
        self.mode_params = {
            0: AHUModeParams(flow=0.0,          mode_name="off"),
            1: AHUModeParams(flow=FLOW_LOW,      mode_name="low"),
            2: AHUModeParams(flow=FLOW_MODERATE,  mode_name="moderate"),
            3: AHUModeParams(flow=FLOW_BOOST,     mode_name="boost"),
        }

    # ------------------------------------------------------------------
    # Zone setpoints — outdoor-temperature-compensated
    # ------------------------------------------------------------------

    def zone_setpoints(self, outdoor_temp: float) -> Tuple[float, float]:
        """Compute zone heating/cooling setpoints from outdoor temperature.

        Both setpoints are linearly interpolated between their low and high
        values over the outdoor temperature range [OUTDOOR_TEMP_LOW, OUTDOOR_TEMP_HIGH].

        Returns:
            (heating_setpoint, cooling_setpoint) in °C.
        """
        t = max(0.0, min(1.0,
            (outdoor_temp - OUTDOOR_TEMP_LOW) / (OUTDOOR_TEMP_HIGH - OUTDOOR_TEMP_LOW)
        ))
        htg = ZONE_HTG_SETPOINT_LOW + t * (ZONE_HTG_SETPOINT_HIGH - ZONE_HTG_SETPOINT_LOW)
        clg = ZONE_CLG_SETPOINT_LOW + t * (ZONE_CLG_SETPOINT_HIGH - ZONE_CLG_SETPOINT_LOW)
        return htg, clg

    # ------------------------------------------------------------------
    # Supply air temperature — return-air-compensated
    # ------------------------------------------------------------------

    def return_air_compensation(self, return_air_temp: float) -> float:
        """Compute supply air temperature setpoint from return air temperature.

        Uses a linear compensation curve between two reference points:
        - return_air <= 21.5 °C  →  supply = 19.0 °C  (building is cool)
        - return_air >= 24.5 °C  →  supply = 17.0 °C  (building is warm)

        Args:
            return_air_temp: Measured return air (plenum) temperature [°C].

        Returns:
            Supply air temperature setpoint [°C].
        """
        if return_air_temp <= RETURN_AIR_TEMP_LOW:
            return SUP_TEMP_AT_LOW
        if return_air_temp >= RETURN_AIR_TEMP_HIGH:
            return SUP_TEMP_AT_HIGH

        # Linear interpolation between the two reference points
        slope = (SUP_TEMP_AT_HIGH - SUP_TEMP_AT_LOW) / (RETURN_AIR_TEMP_HIGH - RETURN_AIR_TEMP_LOW)
        result = SUP_TEMP_AT_LOW + slope * (return_air_temp - RETURN_AIR_TEMP_LOW)
        return max(SUP_TEMP_AT_HIGH, min(SUP_TEMP_AT_LOW, result))

    # ------------------------------------------------------------------
    # Fan flow — schedule + CO2 demand control + extreme cold protection
    # ------------------------------------------------------------------

    def co2_flow_control(
        self, hour: float, day: int, co2_concentration: float, outdoor_temp: float
    ) -> float:
        """Return AHU fan mass flow rate based on schedule, CO2, and outdoor temp.

        Schedule (base flow):
          Workday:  1:30–2:30 low (night flush), 4:30–5:30 moderate (pre-flush),
                    5:30–23:30 low (occupied), otherwise off.
          Weekend:  0:00–2:00 low (night flush), 6:00–17:00 low, otherwise off.

        CO2 demand override (can only increase flow, not decrease):
          <= 500 ppm → no additional flow
          >= 750 ppm → full boost (1.0 kg/s)
          Between   → linear ramp

        Extreme cold cap:
          outdoor <= -25 °C → cap at FLOW_LOW
          outdoor >= -15 °C → no cap
          Between           → linear ramp

        Args:
            hour:               Current simulation hour (fractional, 0–24).
            day:                Day of the week (1 = Sunday … 7 = Saturday).
            co2_concentration:  Max zone CO2 concentration [ppm].
            outdoor_temp:       Outdoor dry-bulb temperature [°C].

        Returns:
            Mass flow rate [kg/s].
        """
        is_workday = day not in (1, 7)  # 1 = Sunday, 7 = Saturday

        # --- Schedule-based base flow ---
        # Removed night flushes (1:30-2:30 workday, 0:00-2:00 weekend) — no occupants, pure waste
        is_pre_flush       = is_workday and 5.5 <= hour < 6.5
        is_weekend_daytime = (not is_workday) and 8.0 <= hour < 16.0
        is_working_hours   = is_workday and 6.5 <= hour < 19.0   # was 5.5-23.5 (18h → 12.5h)

        if is_pre_flush:
            base_flow = FLOW_MODERATE
        elif is_weekend_daytime or is_working_hours:
            base_flow = FLOW_LOW
        else:
            base_flow = 0.0

        # --- CO2 demand override ---
        if co2_concentration <= CO2_MIN_LIMIT:
            co2_flow = 0.0
        elif co2_concentration >= CO2_MAX_LIMIT:
            co2_flow = FLOW_BOOST
        else:
            fraction = (co2_concentration - CO2_MIN_LIMIT) / (CO2_MAX_LIMIT - CO2_MIN_LIMIT)
            co2_flow = FLOW_LOW + fraction * (FLOW_BOOST - FLOW_LOW)

        # CO2 can only increase flow, never decrease it
        target_flow = max(co2_flow, base_flow)

        # --- Extreme cold protection ---
        if outdoor_temp <= OUTDOOR_TEMP_COLD_LIMIT:
            max_allowed = FLOW_LOW
        elif outdoor_temp >= OUTDOOR_TEMP_COLD_UPPER:
            max_allowed = FLOW_MAX
        else:
            slope = (FLOW_MAX - FLOW_LOW) / (OUTDOOR_TEMP_COLD_UPPER - OUTDOOR_TEMP_COLD_LIMIT)
            max_allowed = FLOW_LOW + slope * (outdoor_temp - OUTDOOR_TEMP_COLD_LIMIT)

        return min(target_flow, max_allowed)

    # ------------------------------------------------------------------
    # Main setpoint calculation
    # ------------------------------------------------------------------

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
        """Calculate all control setpoints for the current timestep.

        Args:
            zone_temp:          Representative zone air temperature [°C].
            outdoor_temp:       Outdoor dry-bulb temperature [°C].
            return_air_temp:    AHU return air temperature [°C].
            occupancy:          Current occupancy count (reserved for future use).
            hour:               Current simulation hour (0-24).
            day:                Day of the week (1 = Mon … 7 = Sun).
            co2_concentration:  Maximum zone CO2 concentration [ppm].

        Returns:
            Tuple of:
                - heating_setpoint [°C]
                - cooling_setpoint [°C]
                - supply_air_temp  [°C]
                - fan_flow_rate    [kg/s]
        """

        # --- Supply air temperature (return-air-compensated) ---
        supply_air_temp = self.return_air_compensation(return_air_temp)

        # --- Zone setpoints (outdoor-temperature-compensated) ---
        htg, clg = self.zone_setpoints(outdoor_temp)
        self.htg_setpoint = htg
        self.clg_setpoint = clg

        # --- Fan flow rate (schedule + CO2 boost + cold protection) ---
        flow = self.co2_flow_control(hour, day, co2_concentration, outdoor_temp)

        return self.htg_setpoint, self.clg_setpoint, supply_air_temp, flow
