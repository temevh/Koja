# Strategy: vesa_5

**Owner:** Vesa  
**Approach:** Adaptive S1-tracking RBC + Optuna optimization + auto-saved reproducible best parameters

## What is new

This strategy combines the strongest ideas from earlier `vesa_*` versions and adds
an optimizer workflow focused on reproducibility:

- Uses a robust adaptive rule-based model with occupancy-aware ventilation.
- Optimizer prints every trial and highlights each **new best** immediately.
- On each new best, parameters are written to reproducible files:
  - `best_params.json`
  - `best_params.py`
  - `best_result.txt`
- `run_idf.py` automatically loads `best_params.py`, so reproducing best run is easy.

## Run with current best parameters

```bash
cd strategies/vesa_5
python run_idf.py
```

## Optimize and keep updating best parameters

```bash
cd strategies/vesa_5
python optimize.py
```

Optimization progress is stored in `optuna_out/optuna_study.db`, so you can stop
and continue later.
