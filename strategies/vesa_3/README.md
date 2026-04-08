# Strategy: vesa_3

**Owner:** Vesa  
**Approach:** Deep reinforcement learning (behavioral cloning + SAC), forked from `strategies/drl`.

## Layout

Same files as `drl/`:

| File | Purpose |
|------|---------|
| `train_drl.py` | BC pre-training, SAC fine-tuning, TensorBoard under `models/tb_logs_sac` |
| `evaluate_drl.py` | Load `models/sac_bc_hvac` and run one evaluation year |
| `eplus_sim.py` | Gymnasium `EnergyPlusEnv` — implement `get_reward()` here |
| `variable_config.py` | Sensor / meter / actuator bindings |
| `visualize_output.ipynb` | Cost and comfort analysis from `eplusout.csv` |

Simulation artifacts go to **`vesa_3_output/`** (gitignored). Trained weights go to **`models/`** (gitignored).

## Expert data for BC

By default `train_drl.py` reads expert trajectories from **`../drl/expert_data.json`**. Generate or refresh that file by running an RBC strategy that logs trajectories (e.g. `strategies/rbc_scheduled/run_idf.py`), or copy/link `expert_data.json` into this folder and set `EXPERT_JSON = "expert_data.json"` in `train_drl.py`.

## How to run

From the repo root (with `.venv` activated and dependencies from `requirements.txt`):

```bash
cd strategies/vesa_3
python train_drl.py
python evaluate_drl.py
```

TensorBoard (from `strategies/vesa_3`):

```bash
tensorboard --logdir models/tb_logs_sac
```

## Prerequisites

Python 3.10–3.12, EnergyPlus 25.1/25.2, and `pyenergyplus` via `pip install -r requirements.txt` at project root. See root `README.txt` for full hackathon rules and scoring.

## Colab and performance

- **Paths:** Upload the repo (or clone it) and set `IDF_FILE` / `WEATHER_FILE` / `EXPERT_JSON` in `train_drl.py` to **absolute** paths under `/content/...` (or use `pathlib` / `os.chdir` to your copy of `strategies/vesa_3`).
- **EnergyPlus:** Install the same major version as your `pyenergyplus` wheel (see root `requirements.txt`). Ensure the installer layout matches what `pyenergyplus` expects on Linux.
- **One env at a time:** `eplus_sim.py` is not safe with two live `EnergyPlusEnv` instances. Training uses **CheckpointCallback** instead of `EvalCallback` so a second simulation does not run in parallel with training.
- **Faster dry-runs** (environment variables, optional):

  | Variable | Effect |
  |----------|--------|
  | `VES33_SHORT_DAYS=N` | Episode length ≈ `N` simulated days (+ warm-up buffer) instead of a full year |
  | `VES33_SKIP_BC_EVAL=1` | Skip the full EnergyPlus run after BC (saves a lot of wall time) |
  | `VES33_BC_EPOCHS` | BC epochs (default `50`) |
  | `VES33_SAC_STEPS` | SAC fine-tune steps (default `300000`) |
  | `VES33_SAC_FREEZE_STEPS` | Actor frozen pre-training steps (default `5000`) |
  | `VES33_CHECKPOINT_FREQ` | Target checkpoints spacing (capped relative to episode length) |

- **Bottleneck:** Wall time is dominated by **EnergyPlus**, not SB3. Shorter `VES33_SHORT_DAYS`, skipping BC eval, and a smaller `VES33_SAC_STEPS` smoke test help; a **full-year** train still needs a long run.

## Expert / action scale

`supply_fan_flow` in `ACTION_SPEC` is **0.96 kg/s** max so it matches `strategies/rbc_scheduled` `expert_data.json`. If you regenerate expert with a 1.0 kg/s ceiling, bump the max in `ACTION_SPEC` to `1.0`.
