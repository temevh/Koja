# nibs_bo_002 — Band-Tracking Bayesian Optimization

## Strategy Summary

A fundamentally redesigned HVAC controller that **tracks the actual S1 comfort band** rather than optimizing absolute setpoints. Combined with reactive (sensor-driven) flow control and TPE multivariate optimization over 10 physically meaningful parameters.

### Why this approach?

The nibs_bo_001 analysis revealed:
- **One parameter dominated** (`htg_setpoint_low` = 95.8% importance in fANOVA) because it controlled whether zone temps fell inside/outside the S1 comfort band
- **23 other parameters were noise** — the optimizer burned 150 trials trying to approximate a piecewise-linear comfort band with a single linear ramp
- **Schedule-based flow** had near-zero importance — the fixed schedule was a crude proxy for occupancy

The S1 comfort band is defined by a **piecewise function of 24h rolling outdoor temperature**:
```
S1 lower = 20.5°C                        when t_out_24h ≤ 0°C
         = 20.5 + 0.075 × t_out_24h      when 0 < t ≤ 20°C
         = 22.0°C                         when t > 20°C

S1 upper = 22.0°C                         when t_out_24h ≤ 0°C
         = 22.5 + 0.166 × t_out_24h       when 0 < t ≤ 15°C
         = 25.0°C                          when t > 15°C
```

Trying to fit this with `htg = a + b × outdoor_temp` can never match the breakpoints at 0°C, 15°C, 20°C or the different slopes (0.075 vs 0.166). The band-tracking controller just computes the band directly and places setpoints relative to it.

---

## Architecture

### Band-Tracking Setpoints

The controller maintains a **96-sample ring buffer** (24h × 4 steps/hour) of outdoor temperatures. Each timestep:

1. Push current outdoor temp to ring buffer
2. Compute rolling 24h mean
3. Evaluate the exact S1 band boundaries using the scoring function's formula
4. Set `htg_setpoint = S1_lower + htg_margin`
5. Set `clg_setpoint = S1_upper - clg_margin`

With margins > 0, the controller **cannot violate S1 by construction**. The optimizer controls how aggressively to ride the edges (energy savings) vs. how much buffer to keep (comfort safety).

At night (when unoccupied), a `night_htg_setback` further reduces the heating setpoint to save energy.

### Reactive Flow Control

Replaces the entire time-based schedule with three sensor-reactive mechanisms:

| Mechanism | Trigger | Effect |
|-----------|---------|--------|
| **Occupancy** | `total_occupancy > 0` | Switch between `occupied_flow` and `unoccupied_flow` |
| **CO2 boost** | `CO2 > co2_boost_threshold` | Ramp flow up to `co2_boost_flow` (fixed at 0.9) |
| **Free cooling** | `T_out < T_in - nightflush_delta` AND unoccupied AND `T_in > 22°C` | Apply `nightflush_flow` for passive cooling |

No schedules, no clock-based logic. The controller responds to what it senses.

---

## Parameter Space (10 dimensions)

| # | Parameter | Range | Physical meaning |
|---|-----------|-------|-----------------|
| 1 | `htg_margin` | 0.0 – 2.0 °C | Distance above S1 lower band for heating setpoint |
| 2 | `clg_margin` | 0.0 – 2.0 °C | Distance below S1 upper band for cooling setpoint |
| 3 | `night_htg_setback` | 0.0 – 3.0 °C | Additional heating reduction when unoccupied |
| 4 | `supply_temp_low` | 17 – 21 °C | AHU supply temp when return air is cool |
| 5 | `supply_temp_high` | 16 – 20 °C | AHU supply temp when return air is warm |
| 6 | `occupied_flow` | 0.05 – 0.60 kg/s | Base fan flow when building is occupied |
| 7 | `unoccupied_flow` | 0.0 – 0.30 kg/s | Fan flow when building is empty |
| 8 | `co2_boost_threshold` | 400 – 700 ppm | CO2 level to start ramping up ventilation |
| 9 | `nightflush_delta` | 1.0 – 5.0 °C | Outdoor-indoor differential to trigger free cooling |
| 10 | `nightflush_flow` | 0.2 – 0.8 kg/s | Flow rate during free-cooling ventilation |

### vs. nibs_bo_001

| Aspect | bo_001 | bo_002 |
|--------|--------|--------|
| Parameters | 24 (abstract) | 10 (physical) |
| Setpoint logic | Linear interpolation (mismatches band) | Exact band tracking |
| Flow control | Time schedule + CO2 | Occupancy + CO2 + free cooling |
| Sampler | TPE | TPE multivariate + constant_liar |
| Parallelism | Serial | 8 subprocess workers |

---

## Optimizer

**TPE multivariate** (Tree-structured Parzen Estimator) via Optuna with `constant_liar=True` for parallel safety:
- `multivariate=True` — models parameter correlations (e.g., `htg_margin` + `night_htg_setback`)
- `constant_liar=True` — prevents parallel workers from sampling near in-progress trials
- Thread-safe with `n_jobs > 1` (unlike CMA-ES which had race conditions)
- Sample-efficient at our ~200 trial budget

Warm-started with 3 seed configurations:
- **Conservative**: 0.5°C margins, moderate flow (safe baseline)
- **Aggressive**: 0.1°C margins, minimal flow (energy-saving extreme)
- **Comfort**: 1.0°C margins, high flow (comfort extreme)

### Parallelism

Each trial spawns EnergyPlus as a **subprocess** (`run_trial.py`) for process isolation. Default 8 workers for the i9-10885H's 16 threads. Each sim takes ~45-60s → 200 trials in ~20 min.

---

## Files

| File | Purpose |
|------|---------|
| `bo_model.py` | Band-tracking controller with ring buffer |
| `energyplus_controller.py` | E+ bridge (reads occupancy, passes to model) |
| `optimize.py` | TPE multivariate optimizer, subprocess parallelism |
| `run_trial.py` | Subprocess entry point for single trial |
| `run_idf.py` | EnergyPlus API runner |
| `scoring.py` | Cost computation (energy + CO2 + temp penalties) |
| `evaluate_best.py` | Extract and re-run best trial |

## Usage

```bash
cd strategies/nibs_bo_002/

# Run optimization (200 trials, 8 parallel workers)
python optimize.py --n-trials 200 --n-jobs 8

# Monitor live
optuna-dashboard sqlite:///nibs_bo_002.db

# Evaluate best result
python evaluate_best.py
```
