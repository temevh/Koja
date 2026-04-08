"""Template control model — implement your strategy here.

Your model must implement ``calculate_setpoints()`` with the signature
shown below. The controller calls it every simulation timestep.

Copy this template, rename the file (and class), and implement your
control logic.
"""

from typing import Tuple


class MyModel:
    def __init__(self) -> None:
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
        heating_setpoint = 22.0
        cooling_setpoint = 24.0
        supply_air_temp = 19.0
        fan_flow_rate = 0.5

        return heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow_rate
