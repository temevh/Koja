from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Constants from Vesa's best sweep params
ZONE_HTG_SETPOINT_LOW  = 21.0
ZONE_HTG_SETPOINT_HIGH = 22.0
ZONE_CLG_SETPOINT_LOW  = 23.0
ZONE_CLG_SETPOINT_HIGH = 25.0

OUTDOOR_TEMP_LOW = 0.0          
OUTDOOR_TEMP_HIGH = 20.0        

RETURN_AIR_TEMP_LOW = 21.5      
RETURN_AIR_TEMP_HIGH = 24.5     
SUP_TEMP_AT_LOW = 19.0          
SUP_TEMP_AT_HIGH = 19.0         

CO2_MIN_LIMIT = 600
CO2_MAX_LIMIT = 800

OUTDOOR_TEMP_LOW_LIMIT = -25.0          
OUTDOOR_TEMP_HIGH_LIMIT = -15.0        

FLOW_LOW = 0.15        
FLOW_MODERATE = 0.3   
FLOW_BOOST = 1.0      
FLOW_MAX = 1.0        

@dataclass
class AHUModeParams:
    flow: float
    mode_name: str

class Vesa2Model:
    def __init__(self) -> None:
        self.htg_setpoint: float = ZONE_HTG_SETPOINT_LOW
        self.clg_setpoint: float = ZONE_CLG_SETPOINT_LOW
        self._co2_boosting: bool = False
        self.mode_params = {
            0: AHUModeParams(flow=0.0, mode_name="off"),
            1: AHUModeParams(flow=FLOW_LOW, mode_name="low"),
            2: AHUModeParams(flow=FLOW_MODERATE, mode_name="moderate"),
            3: AHUModeParams(flow=FLOW_BOOST, mode_name="boost"),
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
            base_flow = FLOW_MODERATE
        elif is_nightflush or is_nightflush_weekend or is_weekend_flush or is_working_hours:
            base_flow = FLOW_LOW
        else:
            base_flow = 0.0

        co2_flow = 0.0
        if co2_concentration <= CO2_MIN_LIMIT:
            co2_flow = 0.0
        elif co2_concentration >= CO2_MAX_LIMIT:
            co2_flow = FLOW_BOOST
        else:
            fraction = (co2_concentration - CO2_MIN_LIMIT) / (CO2_MAX_LIMIT - CO2_MIN_LIMIT)
            co2_flow = FLOW_LOW + fraction * (FLOW_BOOST - FLOW_LOW)

        target_flow = max(co2_flow, base_flow)  

        max_allowed = target_flow  
        if outdoortemp <= OUTDOOR_TEMP_LOW_LIMIT:
            max_allowed = FLOW_LOW
        elif outdoortemp >= OUTDOOR_TEMP_HIGH_LIMIT:
            max_allowed = FLOW_MAX
        else:
            slope = (FLOW_MAX - FLOW_LOW) / (OUTDOOR_TEMP_HIGH_LIMIT - OUTDOOR_TEMP_LOW_LIMIT)
            max_allowed = FLOW_LOW + slope * (outdoortemp - OUTDOOR_TEMP_LOW_LIMIT)

        return min(target_flow, max_allowed)

    def return_air_compensation(self, return_air_temp: float) -> float:
        if return_air_temp <= RETURN_AIR_TEMP_LOW:
            return SUP_TEMP_AT_LOW
        if return_air_temp >= RETURN_AIR_TEMP_HIGH:
            return SUP_TEMP_AT_HIGH

        slope = (SUP_TEMP_AT_HIGH - SUP_TEMP_AT_LOW) / (RETURN_AIR_TEMP_HIGH - RETURN_AIR_TEMP_LOW)
        return np.clip(SUP_TEMP_AT_LOW + slope * (return_air_temp - RETURN_AIR_TEMP_LOW), min(SUP_TEMP_AT_LOW, SUP_TEMP_AT_HIGH), max(SUP_TEMP_AT_LOW, SUP_TEMP_AT_HIGH))

    def zone_setpoints(self, outdoor_temp: float) -> Tuple[float, float]:
        t = np.clip((outdoor_temp - OUTDOOR_TEMP_LOW) / (OUTDOOR_TEMP_HIGH - OUTDOOR_TEMP_LOW), 0.0, 1.0)
        htg = ZONE_HTG_SETPOINT_LOW + t * (ZONE_HTG_SETPOINT_HIGH - ZONE_HTG_SETPOINT_LOW)
        clg = ZONE_CLG_SETPOINT_LOW + t * (ZONE_CLG_SETPOINT_HIGH - ZONE_CLG_SETPOINT_LOW)
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
