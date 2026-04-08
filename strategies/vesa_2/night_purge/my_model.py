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
        day: float,
        month: float,
        day_of_month: float,
        co2_concentration: float,
    ) -> Tuple[float, float, float, float]:
        """Calculate control setpoints for the current timestep."""
        
        # --- 1. Zone Temperature Setpoints (S1 limits) ---
        # Temperature limits scale based on outdoor temperature (24-hour rolling Avg in real life, but we use instantaneous to approximate)
        if outdoor_temp <= 0.0:
            heating_setpoint = 20.5
            cooling_setpoint = 22.0
        elif outdoor_temp >= 20.0:
            heating_setpoint = 22.0
            cooling_setpoint = 25.0
            if outdoor_temp >= 15.0 and outdoor_temp < 20.0:
                # Cooling reaches 25 at 15C out, according to README
                pass
        
        # Exact linear interpolations based on S1 limits in README:
        # Heating: 20.5 (at <= 0) to 22.0 (at >= 20) -> slope = 1.5 / 20 = 0.075
        htg_t = max(0.0, min(outdoor_temp, 20.0))
        heating_setpoint = 20.5 + htg_t * 0.075
        
        # Cooling: 22.0 (at <= 0) to 25.0 (at >= 15) -> slope = 3.0 / 15 = 0.2
        clg_t = max(0.0, min(outdoor_temp, 15.0))
        cooling_setpoint = 22.0 + clg_t * 0.2
        
        # --- 2. Supply Air Temperature ---
        # Default AHU supply temp. Can be slightly cooler in summer or just fixed.
        supply_air_temp = 19.0
        if return_air_temp >= 24.5:
            supply_air_temp = 17.0
        elif return_air_temp >= 21.5:
            # Linear ramp from 19 to 17 as return air goes from 21.5 to 24.5
            slope = (17.0 - 19.0) / (24.5 - 21.5)
            supply_air_temp = 19.0 + slope * (return_air_temp - 21.5)

        # --- 3. Fan Flow Rate (Summer Night Purge + CO2 Demand Control) ---
        is_workday = day != 1 and day != 7  # Assuming 1=Sunday, 7=Saturday
        is_working_hours = is_workday and (6.0 <= hour <= 18.0)
        
        # Summer Night Purge Conditions
        is_summer = month in [6, 7, 8]
        # Considered night if not working hours or outside 6 AM to 10 PM. We use occupancy as primary.
        # "nobody is in the building"
        is_unoccupied = occupancy == 0
        purge_temp_condition = 12.0 <= outdoor_temp <= 16.0
        
        # CO2 Demand Control computation
        # Baseline flow during working hours
        base_flow = 0.2 if is_working_hours else 0.0
        
        # Linearly ramp based on CO2 from 500ppm (0% boost) to 750ppm (100% boost -> 1.0 kg/s)
        co2_flow = 0.0
        if co2_concentration > 750.0:
            co2_flow = 1.0
        elif co2_concentration > 500.0:
            fraction = (co2_concentration - 500.0) / (750.0 - 500.0)
            co2_flow = 0.2 + fraction * (1.0 - 0.2)
            
        target_flow = max(base_flow, co2_flow)
        
        # If nobody's there, check if we need to purge
        if is_unoccupied:
            if is_summer and purge_temp_condition and (hour < 6.0 or hour > 20.0):
                # execute summer night purge
                target_flow = 1.0
            else:
                # If unoccupied and not purging (and outside working hours baseline), turn fan off
                if not is_working_hours:
                    target_flow = 0.0
        
        fan_flow_rate = max(0.0, min(1.0, target_flow))
        
        return heating_setpoint, cooling_setpoint, supply_air_temp, fan_flow_rate
