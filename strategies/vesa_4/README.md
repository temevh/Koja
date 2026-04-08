# Strategy: vesa_4

**Owner:** Vesa  
**Approach:** Precision S1-Tracking Adaptive Controller with Bayesian Optimization

## Key idea

This strategy combines the best elements from all hackathon approaches into a single
advanced rule-based controller, then uses Optuna Bayesian optimization to find the
optimal parameter set.

**Novel techniques compared to previous strategies:**

1. **Exact S1 band tracking** using a 24-hour rolling outdoor temperature average
   (matching the official scoring methodology), instead of using instantaneous outdoor temp
2. **Precision setpoint placement** that hugs the S1 comfort band edges with configurable
   margins — minimizes energy by allowing maximum thermal drift within the acceptable range
3. **Occupancy-aware ventilation** using actual per-zone occupancy sensor data to decide
   when ventilation is needed, not just time-of-day schedules
4. **CO2 demand control with hysteresis** to avoid fan cycling when CO2 oscillates
   near thresholds
5. **Pre-conditioning logic** that starts conditioning spaces before work hours to hit
   comfort targets when people arrive
6. **Night setback with S2-floor protection** — relaxes setpoints at night to save energy,
   but never goes below S2 lower limits to avoid penalties
7. **Cold weather flow limiting** that restricts outdoor air during extreme cold
8. **15+ tunable parameters** optimized by Optuna's TPE sampler

## How to run

```bash
cd strategies/vesa_4/
python run_idf.py
```

## How to optimize

```bash
cd strategies/vesa_4/
python optimize.py
```

This runs Optuna Bayesian optimization (100 trials by default). Each trial is one
full-year EnergyPlus simulation (~20s). Press Ctrl+C to stop early — progress is saved.

After optimization, copy the printed best parameters into `run_idf.py`.

## Results

(Run simulation and paste total cost here)
