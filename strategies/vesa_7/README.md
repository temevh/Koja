# Strategy: vesa_7
**Owner:** Vesa  
**Approach:** Two-level supervisory RBC + forecast-driven preconditioning + budget-aware robust control

## How to run
```bash
cd strategies/vesa_7
python run_idf.py
```

## How to optimize (target 7000 EUR)
```bash
cd strategies/vesa_7
python optimize.py --n-trials 150 --target-cost 7000
```

Notes:
- Optimizer auto-saves best parameters to:
  - `best_params.py` (used automatically by `run_idf.py`)
  - `best_params.json`
  - `best_result.txt`
- Optimization state is persisted in `optuna_out/optuna_study.db`, so you can stop and resume.

## Key idea
- **Supervisor layer** updates every 1-2 hours and selects one global mode:
  - `comfort` (occupied stable operation)
  - `economy` (unoccupied energy-saving mode)
  - `purge` (free-cooling ventilation if indoor is warm and outdoor is cooler)
  - `recovery` (preconditioning or emergency recovery)
- **Local layer** runs every timestep (15 min) and maps current mode to:
  - heating/cooling setpoint placement around S1/S2 bands
  - AHU supply temperature target
  - fan flow setpoint with CO2 overlay and cold-weather limiting
- **Forecast-driven preconditioning**:
  - reads dry-bulb forecast from the EPW weather file
  - estimates time to next occupancy from schedule
  - uses forecast min/max + thermal trend to trigger earlier recovery only when needed
- **Budget-aware drift manager (90%-style)**:
  - tracks cumulative comfort/IAQ budget usage over elapsed timesteps
  - allows larger economy drift only when budget pressure is low
  - automatically tightens control and CO2 response when budget pressure rises
- **Robust guardbands**:
  - increases safety margins when forecast spread is large
  - increases margins under high short-term disturbance (rapid zone-temp movement)
  - reduces overfitting to one deterministic trajectory by adapting online

## Notes
- This strategy does not use fixed preflush windows as primary logic.
- Supervisor uses occupancy, CO2 and forecast context to avoid one monolithic rule curve.
- Outputs are written to `strategies/vesa_7/eplus_out/`.
