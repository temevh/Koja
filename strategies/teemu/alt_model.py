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

    def update(self, outdoor_temp):
        self.history.append(outdoor_temp)
        self.t = sum(self.history) / len(self.history)

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
            controller.update(outdoor_temp)
        self.prev_hour = hour

        heating_setpoint = controller.get_lower() + 0.03
        cooling_setpoint = max(heating_setpoint, controller.get_upper() - 0.03)

        # Supply air: proven configuration
        if zone_temp < heating_setpoint + 0.1:
            supply_air_temp = 21.0
        elif zone_temp > cooling_setpoint - 0.1:
            supply_air_temp = 16.0
        else:
            supply_air_temp = float(np.clip(outdoor_temp, 16.0, 21.0))


        # --- Ultra-Granular CO2 Ventilation (2 ppm steps) ---
        # EXPERIMENT: Each flow rate bumped +0.04 vs original baseline
        if co2_concentration >= 760:
            fan_flow_rate = 1.00
        elif co2_concentration >= 758:
            fan_flow_rate = 1.00
        elif co2_concentration >= 756:
            fan_flow_rate = 1.00
        elif co2_concentration >= 754:
            fan_flow_rate = 0.98
        elif co2_concentration >= 752:
            fan_flow_rate = 0.96
        elif co2_concentration >= 750:
            fan_flow_rate = 0.94
        elif co2_concentration >= 748:
            fan_flow_rate = 0.92
        elif co2_concentration >= 746:
            fan_flow_rate = 0.90
        elif co2_concentration >= 744:
            fan_flow_rate = 0.88
        elif co2_concentration >= 742:
            fan_flow_rate = 0.86
        elif co2_concentration >= 740:
            fan_flow_rate = 0.84
        elif co2_concentration >= 738:
            fan_flow_rate = 0.82
        elif co2_concentration >= 736:
            fan_flow_rate = 0.80
        elif co2_concentration >= 734:
            fan_flow_rate = 0.78
        elif co2_concentration >= 732:
            fan_flow_rate = 0.76
        elif co2_concentration >= 730:
            fan_flow_rate = 0.74
        elif co2_concentration >= 728:
            fan_flow_rate = 0.72
        elif co2_concentration >= 726:
            fan_flow_rate = 0.70
        elif co2_concentration >= 724:
            fan_flow_rate = 0.68
        elif co2_concentration >= 722:
            fan_flow_rate = 0.66
        elif co2_concentration >= 720:
            fan_flow_rate = 0.64
        elif co2_concentration >= 718:
            fan_flow_rate = 0.62
        elif co2_concentration >= 716:
            fan_flow_rate = 0.60
        elif co2_concentration >= 714:
            fan_flow_rate = 0.58
        elif co2_concentration >= 712:
            fan_flow_rate = 0.56
        elif co2_concentration >= 710:
            fan_flow_rate = 0.54
        elif co2_concentration >= 708:
            fan_flow_rate = 0.52
        elif co2_concentration >= 706:
            fan_flow_rate = 0.50
        elif co2_concentration >= 704:
            fan_flow_rate = 0.48
        elif co2_concentration >= 702:
            fan_flow_rate = 0.46
        elif co2_concentration >= 700:
            fan_flow_rate = 0.44
        elif co2_concentration >= 698:
            fan_flow_rate = 0.42
        elif co2_concentration >= 696:
            fan_flow_rate = 0.40
        elif co2_concentration >= 694:
            fan_flow_rate = 0.38
        elif co2_concentration >= 692:
            fan_flow_rate = 0.36
        elif co2_concentration >= 690:
            fan_flow_rate = 0.34
        elif co2_concentration >= 688:
            fan_flow_rate = 0.32
        elif co2_concentration >= 686:
            fan_flow_rate = 0.30
        elif co2_concentration >= 684:
            fan_flow_rate = 0.28
        elif co2_concentration >= 682:
            fan_flow_rate = 0.26
        elif co2_concentration >= 680:
            fan_flow_rate = 0.24
        elif co2_concentration >= 676:
            fan_flow_rate = 0.22
        elif co2_concentration >= 672:
            fan_flow_rate = 0.20
        elif co2_concentration >= 668:
            fan_flow_rate = 0.18
        elif co2_concentration >= 664:
            fan_flow_rate = 0.16
        elif co2_concentration >= 660:
            fan_flow_rate = 0.14
        elif co2_concentration >= 654:
            fan_flow_rate = 0.12
        elif co2_concentration >= 648:
            fan_flow_rate = 0.10
        elif co2_concentration >= 642:
            fan_flow_rate = 0.08
        elif co2_concentration >= 636:
            fan_flow_rate = 0.07
        elif co2_concentration >= 630:
            fan_flow_rate = 0.06
        else:
            fan_flow_rate = 0.00

        return heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow_rate