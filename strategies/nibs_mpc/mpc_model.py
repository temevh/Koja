"""MPC controller model — main interface for EnergyPlus.

Implements calculate_setpoints() as required by the controller template.
Uses a trained dynamics ensemble + CEM solver to optimize actions at each
timestep over a short planning horizon.

Also supports update_state() to receive full per-zone observations from
the extended energyplus_controller.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Add strategy root to path for mpc/ and train/ imports
_STRATEGY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_STRATEGY_DIR))

from mpc.cem_solver import CEMSolver
from train.dynamics_model import DynamicsEnsemble

MODELS_DIR = _STRATEGY_DIR / "models"
RING_BUFFER_SIZE = 96  # 24h at 15-min timesteps


def _load_surrogate(models_dir: Path, backend: str):
    """Load either NN ensemble or LightGBM surrogate."""
    if backend == "lgbm":
        from train.lgbm_model import LightGBMDynamics
        return LightGBMDynamics(models_dir)
    return DynamicsEnsemble(models_dir)


class MPCModel:
    """Model Predictive Controller using learned dynamics + CEM."""

    _LGBM_DEFAULTS = dict(horizon=4, n_samples=200, n_elite=20, n_iterations=3)
    _FAST_DEFAULTS = dict(horizon=15, n_samples=200, n_elite=10, n_iterations=1)
    # _FAST_DEFAULTS = dict(horizon=2, n_samples=120, n_elite=15, n_iterations=1)


    def __init__(
        self,
        models_dir: str | Path = MODELS_DIR,
        horizon: int | None = None,
        n_samples: int | None = None,
        n_elite: int | None = None,
        n_iterations: int | None = None,
        backend: str = "nn",
        fast: bool = False,
        replan_interval: int = 1,
    ) -> None:
        # Apply backend-specific defaults
        if fast:
            defaults = self._FAST_DEFAULTS
            replan_interval = max(replan_interval, 4)  # hold action for 4 steps
        elif backend == "lgbm":
            defaults = self._LGBM_DEFAULTS
        else:
            defaults = dict(horizon=6, n_samples=400, n_elite=40, n_iterations=5)

        self.ensemble = _load_surrogate(Path(models_dir), backend)
        self.solver = CEMSolver(
            horizon=horizon or defaults["horizon"],
            n_samples=n_samples or defaults["n_samples"],
            n_elite=n_elite or defaults["n_elite"],
            n_iterations=n_iterations or defaults["n_iterations"],
        )

        # Replan interval: hold actions for N timesteps between solves
        self._replan_interval = replan_interval
        self._cached_action: Optional[Tuple[float, float, float, float]] = None

        # Outdoor temp ring buffer for 24h rolling mean
        self._outdoor_buffer: deque = deque(maxlen=RING_BUFFER_SIZE)

        # Full observation dict (set by controller via update_state)
        self._obs: Optional[dict] = None

        # Step counter for diagnostics
        self._step = 0

    def _update_outdoor_rolling(self, outdoor_temp: float) -> float:
        self._outdoor_buffer.append(outdoor_temp)
        return sum(self._outdoor_buffer) / len(self._outdoor_buffer)

    def update_state(self, obs: dict) -> None:
        """Receive full per-zone observations from the controller."""
        self._obs = obs

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
        """Compute optimal setpoints via CEM planning."""
        t_out_24h = self._update_outdoor_rolling(outdoor_temp)
        self._step += 1

        # Skip CEM if within replan interval and we have a cached action
        if (self._cached_action is not None
                and self._step % self._replan_interval != 1):
            return self._cached_action

        is_occupied = occupancy > 0.5

        # Build state vector from full per-zone obs (if available)
        if self._obs is not None:
            zone_temps = np.array([self._obs[f"space{i}_temp"] for i in range(1, 6)])
            zone_co2 = np.array([self._obs[f"space{i}_co2"] for i in range(1, 6)])
        else:
            # Fallback: replicate aggregate values across zones
            zone_temps = np.full(5, zone_temp)
            zone_co2 = np.full(5, co2_concentration)

        # State vector: [zone_temps(5), zone_co2(5), outdoor_temp, plenum_temp]
        current_state = np.concatenate([
            zone_temps,
            zone_co2,
            [outdoor_temp, return_air_temp],
        ])

        # Time features
        time_features = np.array([
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * day / 7.0),
            np.cos(2 * np.pi * day / 7.0),
        ])

        # Run CEM
        action, cost = self.solver.solve(
            current_state, self.ensemble, t_out_24h, time_features,
        )

        htg, clg, supply_temp, flow = action

        if self._step % 96 == 0:  # Log once per day
            print(f"  MPC step {self._step}: htg={htg:.1f} clg={clg:.1f} "
                  f"supply={supply_temp:.1f} flow={flow:.2f} "
                  f"cost={cost:.3f}€ t_out_24h={t_out_24h:.1f}°C")

        result = (float(htg), float(clg), float(supply_temp), float(flow))
        self._cached_action = result
        return result
