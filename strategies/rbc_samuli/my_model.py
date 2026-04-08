"""Template control model — implement your strategy here.

Your model must implement ``calculate_setpoints()`` with the signature
shown below. The controller calls it every simulation timestep.

Copy this template, rename the file (and class), and implement your
control logic.
"""

from typing import Tuple
import numpy as np
from collections import deque

class TemperatureTargeter:
    def __init__(self):
        self.history = deque(maxlen=24)
        self.t = 0

    def update(self, outdoor_temp):
        self.history.append(outdoor_temp)
        self.t = sum(self.history) / len(self.history)

    def get_lower(self):
        t = self.t

        lower = 20.5 if t <= 0 else (20.5 + 0.075 * t if t <= 20 else 22.0)

        return lower   
    
    def get_upper(self):
        t = self.t

        upper = 22.0 if t <= 0 else (22.5 + 0.166 * t if t <= 15 else 25.0)

        return upper   
    
    def get_target(self):
        lower = self.get_lower(self)
        upper = self.get_upper(self)

        return (lower + upper) / 2
    
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
        # TODO: Implement your control logic here

        if hour != self.prev_hour:
            controller.update(outdoor_temp)
        self.prev_hour = hour

        heating_setpoint = controller.get_lower() + 0.7
        cooling_setpoint = max(heating_setpoint, controller.get_upper() - 0.7)

        supply_air_temp = 18
        if co2_concentration < CO2_THRESHOLDS[0] - 100:
            fan_flow_rate = 0.1
        else:
            fan_flow_rate = 1.0
        return heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow_rate
