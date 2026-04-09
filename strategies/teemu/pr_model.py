from typing import Tuple
from collections import deque
import numpy as np

class TemperatureTargeter:
    def __init__(self):
        # 24-hour history for moving average calculation
        self.history = deque(maxlen=24)
        self.t = 0
        self.prev_co2_ = 0
        self.co2 = 0

    def update(self, outdoor_temp, co2_concentration):
        # Add current outdoor temp to history and update moving average (self.t)
        self.history.append(outdoor_temp)
        self.t = sum(self.history) / len(self.history)
        self.co2 = co2_concentration

    def get_lower(self):
        # Calculates lower temperature bound based on outdoor average
        t = self.t
        lower = 20.5 if t <= 0 else (20.5 + 0.075 * t if t <= 20 else 22.0)
        # Default to 22.0 during startup/insufficient history
        if len(self.history) < 23:
            lower = 22
        return lower   
    
    def get_upper(self):
        # Calculates upper temperature bound based on outdoor average
        t = self.t
        upper = 22.0 if t <= 0 else (22.5 + 0.166 * t if t <= 15 else 25.0)
        # Default to 25.0 during startup/insufficient history
        if len(self.history) < 23:
            upper = 25
        return upper   
    
    def get_target(self):
        # Helper to get the midpoint between bounds
        lower = self.get_lower()
        upper = self.get_upper()
        return (lower + upper) / 2
    
    def update_post(self):
        # Store current CO2 for reference in next timestep
        self.prev_co2_ = self.co2

    def prev_co2(self):
        return self.prev_co2_
    
CO2_THRESHOLDS = [770, 970, 1220]

controller = TemperatureTargeter()

class MyModel:
    def __init__(self) -> None:
        self.prev_hour = 0
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
        """Calculate control setpoints for the current timestep.

        Args:
            zone_temp:          Average zone air temperature [°C].
            outdoor_temp:       Outdoor dry-bulb temperature [°C].
            return_air_temp:    AHU return (plenum) air temperature [°C].
            occupancy:          Current occupancy count.
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

        # Update outdoor temp history once per hour
        if hour != self.prev_hour:
            controller.update(outdoor_temp, co2_concentration)
        self.prev_hour = hour

        # heating and cooling margins
        heating_setpoint = controller.get_lower() + 0.0275
        cooling_setpoint = max(heating_setpoint, controller.get_upper() - 0.0085)
        
        # Determine supply air temperature based on current zone condition
        if zone_temp < heating_setpoint + 0.1:
            supply_air_temp = 21.0  # Maximum heat allowed (uses Heating Coil)
        elif zone_temp > cooling_setpoint - 0.1:
            supply_air_temp = 16.0  # Maximum cooling allowed (uses Cooling Coil)
        else:
            # Neutral zone: use outdoor air temperature capped at equipment limits
            supply_air_temp = float(np.clip(outdoor_temp, 16.0, 21.0))

        """
         --- Balanced Ventilaton Ramp ---
         Gradually increase fan speed as CO2 approaches the 770 ppm penalty wall.
         This staircase optimizes the cubic power law (Flow^3) to minimize energy.

         Ugly ass if-elif checks but it works and could be optimized by adding more steps or coming up with an actual function
         After the hackathon we set up a simulation/script that goes through different combinations, will
         update the new numbers at the top of this file as comments if we find anything useful
         
         Currnt solution also most likely overventilates, due to 0 CO2 penalty
         Reducing overventilating would bring the energy costs down
        """
        if co2_concentration >= CO2_THRESHOLDS[0] - 15: # 755 ppm
            fan_flow_rate = 0.94
        elif co2_concentration >= CO2_THRESHOLDS[0] - 25: # 745 ppm
            fan_flow_rate = 0.93
        elif co2_concentration >= CO2_THRESHOLDS[0] - 30: # 740 ppm
            fan_flow_rate = 0.87
        elif co2_concentration >= CO2_THRESHOLDS[0] - 35: # 735 ppm
            fan_flow_rate = 0.80
        elif co2_concentration >= CO2_THRESHOLDS[0] - 40: # 730 ppm
            fan_flow_rate = 0.80
        elif co2_concentration >= CO2_THRESHOLDS[0] - 45: # 725 ppm
            fan_flow_rate = 0.77
        elif co2_concentration >= CO2_THRESHOLDS[0] - 50: # 720 ppm
            fan_flow_rate = 0.75
        elif co2_concentration >= CO2_THRESHOLDS[0] - 55: # 715 ppm
            fan_flow_rate = 0.70
        elif co2_concentration >= CO2_THRESHOLDS[0] - 60: # 710 ppm
            fan_flow_rate = 0.65
        elif co2_concentration >= CO2_THRESHOLDS[0] - 65: # 705 ppm
            fan_flow_rate = 0.60
        elif co2_concentration >= CO2_THRESHOLDS[0] - 70: # 700 ppm
            fan_flow_rate = 0.55
        elif co2_concentration >= CO2_THRESHOLDS[0] - 75: # 695 ppm
            fan_flow_rate = 0.53
        elif co2_concentration >= CO2_THRESHOLDS[0] - 80: # 690 ppm
            fan_flow_rate = 0.50
        elif co2_concentration >= CO2_THRESHOLDS[0] - 85: # 685 ppm
            fan_flow_rate = 0.45
        elif co2_concentration >= CO2_THRESHOLDS[0] - 90: # 680 ppm
            fan_flow_rate = 0.40
        elif co2_concentration >= CO2_THRESHOLDS[0] - 95: # 675 ppm
            fan_flow_rate = 0.3
        elif co2_concentration >= CO2_THRESHOLDS[0] - 100: # 670 ppm
            fan_flow_rate = 0.20
        elif co2_concentration >= CO2_THRESHOLDS[0] - 110: # 660 ppm
            fan_flow_rate = 0.20
        elif co2_concentration >= CO2_THRESHOLDS[0] - 120: # 650 ppm
            fan_flow_rate = 0.2
        elif co2_concentration >= CO2_THRESHOLDS[0] - 250: # 630 ppm
            fan_flow_rate = 0.0
        else:
            fan_flow_rate = 0.0
            
        controller.update_post()

        return heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow_rate