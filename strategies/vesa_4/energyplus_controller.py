"""Enhanced EnergyPlus runtime controller for vesa_4 strategy.

Extends the template controller with:
- Per-zone occupancy data for occupancy-aware control
- Month for seasonal logic
- Total occupancy sum for the model
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.variable_config import METERS, VARIABLES, ACTUATORS


class EnergyPlusController:
    def __init__(self, api: Any, model: Any) -> None:
        self.api = api
        self.model = model
        self.handles: Dict[str, int] = {}

    def initialize_handles(self, state: Any) -> None:
        ex = self.api.exchange

        for name, (var, key) in VARIABLES.items():
            self.handles[name] = ex.get_variable_handle(state, var, key)

        for name, (ctype, control, key) in ACTUATORS.items():
            self.handles[name] = ex.get_actuator_handle(state, ctype, control, key)

        for name, meter_name in METERS.items():
            self.handles[name] = ex.get_meter_handle(state, meter_name)

        bad = [n for n, h in self.handles.items() if h == -1]
        if bad:
            print(f"WARNING: the following handles resolved to -1: {bad}")
        else:
            print(f"All {len(self.handles)} handles initialised successfully.")

    def get_variable(self, name: str, state: Any, default: float = 0.0) -> float:
        handle = self.handles.get(name)
        if handle is None or handle == -1:
            return default
        if name in METERS:
            return self.api.exchange.get_meter_value(state, handle)
        return self.api.exchange.get_variable_value(state, handle)

    def set_actuator(self, name: str, value: float, state: Any) -> None:
        handle = self.handles.get(name)
        if handle is not None and handle != -1:
            self.api.exchange.set_actuator_value(state, handle, value)

    def _collect_obs(self, state: Any) -> Dict:
        obs: Dict[str, float] = {
            "outdoor_temp": self.get_variable("outdoor_temp", state),
            "plenum_temp": self.get_variable("plenum_temp", state),
        }

        for i in range(1, 6):
            obs[f"space{i}_temp"] = self.get_variable(f"space{i}_temp", state)
            obs[f"space{i}_rh"] = self.get_variable(f"space{i}_rh", state)
            obs[f"space{i}_co2"] = self.get_variable(f"space{i}_co2", state)
            obs[f"space{i}_occupancy"] = self.get_variable(f"space{i}_occupancy", state)

        obs["electricity_hvac"] = self.get_variable("electricity_hvac", state)
        obs["gas_total"] = self.get_variable("gas_total", state)
        obs["hour"] = float(self.api.exchange.hour(state))
        obs["day_of_week"] = float(self.api.exchange.day_of_week(state))
        obs["month"] = float(self.api.exchange.month(state))

        return obs

    def control_callback(self, state: Any) -> None:
        if not self.handles:
            return

        obs = self._collect_obs(state)

        temps = [obs[f"space{i}_temp"] for i in range(1, 6)]
        avg_temp = sum(temps) / len(temps)

        co2s = [obs[f"space{i}_co2"] for i in range(1, 6)]
        max_co2 = max(co2s)

        total_occupancy = sum(obs[f"space{i}_occupancy"] for i in range(1, 6))

        htg, clg, supply_air_temp, flow = self.model.calculate_setpoints(
            zone_temp=avg_temp,
            outdoor_temp=obs["outdoor_temp"],
            return_air_temp=obs["plenum_temp"],
            occupancy=total_occupancy,
            hour=obs["hour"],
            day=obs["day_of_week"],
            co2_concentration=max_co2,
            month=obs["month"],
        )

        self.set_actuator("htg_setpoint", htg, state)
        self.set_actuator("clg_setpoint", clg, state)
        self.set_actuator("ahu_temperature_setpoint", supply_air_temp, state)
        self.set_actuator("ahu_mass_flow_rate_setpoint", flow, state)

    def get_api_data(self, state: Any) -> None:
        api_data = self.api.exchange.get_api_data(state)
        with open("api_initialization_log.txt", "w", encoding="utf-8") as f:
            for item in api_data:
                f.write(
                    f"{item.name}, key: {item.key}, type: {item.type}, "
                    f"{item.unit}, {item.what}\n"
                )
        print(f"API data ({len(api_data)} items) written to api_initialization_log.txt")
