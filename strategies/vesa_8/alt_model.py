"""Template control model — implement your strategy here.

Your model must implement ``calculate_setpoints()`` with the signature
shown below. The controller calls it every simulation timestep.

Copy this template, rename the file (and class), and implement your
control logic.
"""

from typing import Tuple

from collections import deque
import numpy as np

class TemperatureTargeter:
    def __init__(self):
        self.history = deque(maxlen=24)
        self.t = 0
        self.prev_co2_ = 0
        self.co2 = 0

    def update(self, outdoor_temp, co2_concentration):
        self.history.append(outdoor_temp)
        self.t = sum(self.history) / len(self.history)
        self.co2 = co2_concentration

    def get_lower(self):
        t = self.t

        lower = 20.5 if t <= 0 else (20.5 + 0.075 * t if t <= 20 else 22.0)
        if len(self.history) < 24:
            lower = 22
        return lower   
    
    def get_upper(self):
        t = self.t

        upper = 22.0 if t <= 0 else (22.5 + 0.166 * t if t <= 15 else 25.0)
        if len(self.history) < 24:
            upper = 25
        return upper   
    
    def get_target(self):
        lower = self.get_lower(self)
        upper = self.get_upper(self)

        return (lower + upper) / 2
    
    def update_post(self):
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

        if hour != self.prev_hour:
            controller.update(outdoor_temp, co2_concentration)
        self.prev_hour = hour

        heating_setpoint = controller.get_lower() + 0.03
        cooling_setpoint = max(heating_setpoint, controller.get_upper() - 0.03)
        #print(f"heating: {heating_setpoint}, cooling: {cooling_setpoint}")
        #supply_air_temp = 18.0
        if zone_temp < heating_setpoint + 0.1:
            supply_air_temp = 21.0  # Max allowed (Heat with Gas)
        elif zone_temp > cooling_setpoint - 0.1:
            supply_air_temp = 16.0  # Min allowed (Cooling)
        else:
            # Optimize neutral state: do not use Boiler to heat to 18.5 in winter if we can just supply 16.0!

            supply_air_temp = float(np.clip(outdoor_temp, 16.0, 21.0))


        if co2_concentration >= CO2_THRESHOLDS[0] - 15: # 750
            fan_flow_rate = 0.96
        elif co2_concentration >= CO2_THRESHOLDS[0] - 25: # 745
            fan_flow_rate = 0.93
        elif co2_concentration >= CO2_THRESHOLDS[0] - 30: # 740
            fan_flow_rate = 0.87
        elif co2_concentration >= CO2_THRESHOLDS[0] - 35: # 735
            fan_flow_rate = 0.80
        elif co2_concentration >= CO2_THRESHOLDS[0] - 40: # 730
            fan_flow_rate = 0.79
        elif co2_concentration >= CO2_THRESHOLDS[0] - 45: # 725
            fan_flow_rate = 0.75
        elif co2_concentration >= CO2_THRESHOLDS[0] - 50: # 720
            fan_flow_rate = 0.72
        elif co2_concentration >= CO2_THRESHOLDS[0] - 55: # 715
            fan_flow_rate = 0.65
        elif co2_concentration >= CO2_THRESHOLDS[0] - 60: # 710
            fan_flow_rate = 0.50
        elif co2_concentration >= CO2_THRESHOLDS[0] - 65: # 705
            fan_flow_rate = 0.55
        elif co2_concentration >= CO2_THRESHOLDS[0] - 70: # 700
            fan_flow_rate = 0.50
        elif co2_concentration >= CO2_THRESHOLDS[0] - 75: # 695
            fan_flow_rate = 0.45
        elif co2_concentration >= CO2_THRESHOLDS[0] - 80: # 690
            fan_flow_rate = 0.40
        elif co2_concentration >= CO2_THRESHOLDS[0] - 85: # 685
            fan_flow_rate = 0.35
        elif co2_concentration >= CO2_THRESHOLDS[0] - 90: # 680
            fan_flow_rate = 0.50
        elif co2_concentration >= CO2_THRESHOLDS[0] - 95: # 675
            fan_flow_rate = 0.5
        elif co2_concentration >= CO2_THRESHOLDS[0] - 100: # 670
            fan_flow_rate = 0.20
        elif co2_concentration >= CO2_THRESHOLDS[0] - 110: # 660
            fan_flow_rate = 0.40
        elif co2_concentration >= CO2_THRESHOLDS[0] - 120: # 650
            fan_flow_rate = 0.2
        elif co2_concentration >= CO2_THRESHOLDS[0] - 250: # 630
            fan_flow_rate = 0.0
        else:
            fan_flow_rate = 0.0


        controller.update_post()

        return heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow_rate