# Strategy: vesa_6
**Owner:** Vesa  
**Approach:** Hybrid RBC using S1-band tracking, return-air supply compensation, and CO2-reactive ventilation with preflush/night-flush

## How to run
```bash
cd strategies/vesa_6
python run_idf.py
```

Outputs are written to `strategies/vesa_6/eplus_out/`.

## Automatic optimization
```bash
cd strategies/vesa_6
python optimize.py
```

- Progress and study DB are saved in `optuna_out/optuna_study.db`.
- Best parameters are auto-written to:
  - `best_params.py`
  - `best_params.json`
  - `best_result.txt`
- `run_idf.py` always loads `BEST_PARAMS` from `best_params.py`.

## Key idea
- Track heating/cooling setpoints from the 24-hour rolling outdoor mean (S1 comfort band logic from scoring rules).
- Use return-air temperature to interpolate AHU supply temperature in allowed range.
- Combine schedule base flow + occupancy + CO2 boost.
- Add preflush before workday and free-cooling night flush when indoor is warm and outdoor is cooler.
- Cap fan flow in very cold weather to avoid unnecessary ventilation heat loss.

## Notes
- Actuator limits and ordering follow `README.txt` constraints:
  - heating/cooling setpoints clamped to 18...25 C
  - AHU supply temperature clamped to 16...21 C
  - fan flow clamped to 0...1 kg/s
  - heating setpoint always <= cooling setpoint
