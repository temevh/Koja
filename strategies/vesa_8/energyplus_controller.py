"""EnergyPlus runtime controller for vesa_8."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path so we can import shared config.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.variable_config import ACTUATORS, METERS, VARIABLES


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
        obs = {
            "outdoor_temp": self.get_variable("outdoor_temp", state),
            "plenum_temp": self.get_variable("plenum_temp", state),
            "space1_temp": self.get_variable("space1_temp", state),
            "space1_co2": self.get_variable("space1_co2", state),
            "space2_temp": self.get_variable("space2_temp", state),
            "space2_co2": self.get_variable("space2_co2", state),
            "space3_temp": self.get_variable("space3_temp", state),
            "space3_co2": self.get_variable("space3_co2", state),
            "space4_temp": self.get_variable("space4_temp", state),
            "space4_co2": self.get_variable("space4_co2", state),
            "space5_temp": self.get_variable("space5_temp", state),
            "space5_co2": self.get_variable("space5_co2", state),
            "hour": float(self.api.exchange.hour(state)),
            "day_of_week": float(self.api.exchange.day_of_week(state)),
        }
        return obs

    def control_callback(self, state: Any) -> None:
        if not self.handles:
            return

        obs = self._collect_obs(state)
        avg_temp = sum(obs[f"space{i}_temp"] for i in range(1, 6)) / 5.0
        max_co2 = max(obs[f"space{i}_co2"] for i in range(1, 6))

        htg, clg, supply_air_temp, flow = self.model.calculate_setpoints(
            zone_temp=avg_temp,
            outdoor_temp=obs["outdoor_temp"],
            return_air_temp=obs["plenum_temp"],
            occupancy=0.0,
            hour=obs["hour"],
            day=int(obs["day_of_week"]),
            co2_concentration=max_co2,
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
