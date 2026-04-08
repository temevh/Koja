"""EnergyPlus runtime controller for MPC data collection and evaluation.

Extends the template controller to:
  - Pass full per-zone observations to the model via update_state()
  - Optionally log complete trajectory (state + action) to CSV
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.variable_config import METERS, VARIABLES, ACTUATORS


class EnergyPlusController:
    def __init__(self, api: Any, model: Any, log_trajectory: bool = False) -> None:
        self.api = api
        self.model = model
        self.handles: Dict[str, int] = {}
        self.log_trajectory = log_trajectory
        self._trajectory: List[Dict] = []

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

        # Pass full per-zone state to model (for MPC planning)
        if hasattr(self.model, "update_state"):
            self.model.update_state(obs)

        htg, clg, supply_air_temp, flow = self.model.calculate_setpoints(
            zone_temp=avg_temp,
            outdoor_temp=obs["outdoor_temp"],
            return_air_temp=obs["plenum_temp"],
            occupancy=total_occupancy,
            hour=obs["hour"],
            day=int(obs["day_of_week"]),
            co2_concentration=max_co2,
        )

        self.set_actuator("htg_setpoint", htg, state)
        self.set_actuator("clg_setpoint", clg, state)
        self.set_actuator("ahu_temperature_setpoint", supply_air_temp, state)
        self.set_actuator("ahu_mass_flow_rate_setpoint", flow, state)

        if self.log_trajectory:
            row = dict(obs)
            row["action_htg"] = htg
            row["action_clg"] = clg
            row["action_supply_temp"] = supply_air_temp
            row["action_fan_flow"] = flow
            self._trajectory.append(row)

    def save_trajectory(self, path: str | Path) -> None:
        """Write collected trajectory to CSV."""
        if not self._trajectory:
            print("WARNING: no trajectory data to save.")
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._trajectory[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._trajectory)
        print(f"Saved trajectory ({len(self._trajectory)} rows) → {path}")
