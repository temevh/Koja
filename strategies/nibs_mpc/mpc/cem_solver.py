"""Cross-Entropy Method (CEM) solver for MPC action optimization.

Solves at each timestep:
    minimize Σ_{k=0}^{H-1} cost(predicted_state_k, action_k)
    over action_0, ..., action_{H-1}
    subject to physical bounds and htg ≤ clg

Uses iterative Gaussian refinement: sample → evaluate → refit to elites.
Supports warm-starting from previous solution for temporal consistency.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mpc.cost_function import compute_step_cost_batch


# Action bounds: [htg, clg, supply_temp, fan_flow]
ACTION_LOW = np.array([18.0, 18.0, 16.0, 0.0])
ACTION_HIGH = np.array([25.0, 25.0, 21.0, 1.0])
ACTION_DIM = 4


class CEMSolver:
    """Cross-Entropy Method for short-horizon action optimization."""

    def __init__(
        self,
        horizon: int = 6,
        n_samples: int = 400,
        n_elite: int = 40,
        n_iterations: int = 5,
        initial_std: float = 1.0,
        min_std: float = 0.05,
    ) -> None:
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations
        self.initial_std = initial_std
        self.min_std = min_std

        # Warm-start cache
        self._prev_mean: Optional[np.ndarray] = None  # (H, 4)

    def _enforce_constraints(self, actions: np.ndarray) -> np.ndarray:
        """Clip to bounds and enforce htg <= clg.

        Args:
            actions: shape (n_samples, horizon, 4)
        """
        # Clip to physical bounds
        actions = np.clip(actions, ACTION_LOW, ACTION_HIGH)

        # Enforce htg <= clg
        htg = actions[..., 0]
        clg = actions[..., 1]
        mask = htg > clg
        mid = (htg + clg) / 2.0
        actions[..., 0] = np.where(mask, mid, htg)
        actions[..., 1] = np.where(mask, mid, clg)

        return actions

    def _rollout_cost(
        self,
        actions: np.ndarray,
        current_state: np.ndarray,
        ensemble,
        t_out_24h: float,
        time_features: np.ndarray,
    ) -> np.ndarray:
        """Evaluate total cost for a batch of action sequences.

        Args:
            actions: (n_samples, horizon, 4)
            current_state: (12,) — [zone_temps(5), zone_co2(5), outdoor_temp, plenum_temp]
            ensemble: DynamicsEnsemble instance
            t_out_24h: 24h rolling mean outdoor temp
            time_features: (4,) — [sin_hour, cos_hour, sin_day, cos_day]

        Returns:
            (n_samples,) — total cost per trajectory
        """
        N = actions.shape[0]
        H = actions.shape[1]

        # State: current physical state replicated for all samples
        state = np.tile(current_state, (N, 1))  # (N, 12)
        total_cost = np.zeros(N)

        # Time features: assume constant over horizon (small error for 1.5h)
        time_feat = np.tile(time_features, (N, 1))  # (N, 4)

        for k in range(H):
            act_k = actions[:, k, :]  # (N, 4)

            # Build model input: [state(12), time(4), action(4)] = 20
            model_input = np.hstack([state, time_feat, act_k])  # (N, 20)

            # Predict deltas + energy
            pred = ensemble.predict(model_input)  # (N, 12)
            delta_temps = pred[:, :5]   # (N, 5)
            delta_co2 = pred[:, 5:10]   # (N, 5)
            step_elec = pred[:, 10]     # (N,)
            step_gas = pred[:, 11]      # (N,)

            # Energy can't be negative
            step_elec = np.maximum(step_elec, 0.0)
            step_gas = np.maximum(step_gas, 0.0)

            # Current zone states (for cost computation — before update)
            zone_temps = state[:, :5]  # (N, 5)
            zone_co2 = state[:, 5:10]  # (N, 5)

            # Next state via delta
            next_temps = zone_temps + delta_temps
            next_co2 = np.clip(zone_co2 + delta_co2, 400.0, 5000.0)

            # Cost at this step (evaluate on predicted next state)
            step_cost = compute_step_cost_batch(
                next_temps, next_co2, step_elec, step_gas, t_out_24h,
            )
            total_cost += step_cost

            # Update state for next horizon step
            state = state.copy()
            state[:, :5] = next_temps
            state[:, 5:10] = next_co2
            # outdoor_temp and plenum_temp assumed constant over horizon

        return total_cost

    def solve(
        self,
        current_state: np.ndarray,
        ensemble,
        t_out_24h: float,
        time_features: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Solve for optimal first action via CEM.

        Args:
            current_state: (12,) — [zone_temps(5), zone_co2(5), outdoor_temp, plenum_temp]
            ensemble: DynamicsEnsemble instance
            t_out_24h: 24h rolling mean outdoor temp
            time_features: (4,) — [sin_hour, cos_hour, sin_day, cos_day]

        Returns:
            (best_first_action, best_cost) where best_first_action is (4,)
        """
        H = self.horizon
        N = self.n_samples

        # Initialize mean: warm-start or center of action range
        if self._prev_mean is not None:
            # Shift previous solution left by 1 step
            mean = np.zeros((H, ACTION_DIM))
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = self._prev_mean[-1]  # repeat last action
        else:
            mean = np.tile((ACTION_LOW + ACTION_HIGH) / 2.0, (H, 1))

        # Initialize std
        std = np.full((H, ACTION_DIM), self.initial_std)

        best_action = mean[0].copy()
        best_cost = float("inf")

        for it in range(self.n_iterations):
            # Sample action sequences: (N, H, 4)
            noise = np.random.randn(N, H, ACTION_DIM) * std[np.newaxis, :, :]
            actions = mean[np.newaxis, :, :] + noise
            actions = self._enforce_constraints(actions)

            # Evaluate costs
            costs = self._rollout_cost(
                actions, current_state, ensemble, t_out_24h, time_features,
            )

            # Select elites
            elite_idx = np.argsort(costs)[:self.n_elite]
            elites = actions[elite_idx]  # (n_elite, H, 4)
            elite_costs = costs[elite_idx]

            # Refit Gaussian
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), self.min_std)

            # Track best
            if elite_costs[0] < best_cost:
                best_cost = elite_costs[0]
                best_action = elites[0, 0].copy()

        # Cache for warm-start
        self._prev_mean = mean.copy()

        return best_action, float(best_cost)

    def reset(self) -> None:
        """Clear warm-start cache (e.g., at start of new simulation)."""
        self._prev_mean = None
