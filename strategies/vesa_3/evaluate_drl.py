"""
evaluate_drl.py — vesa_3: load a trained SAC model and run full-year evaluation.

Uses OBS_SPEC / ACTION_SPEC from train_drl in this folder.
"""

import os

import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from eplus_sim import EnergyPlusEnv
from train_drl import OBS_SPEC, ACTION_SPEC, _build_config


# =========================================================================
# PATHS — edit these to match your setup
# =========================================================================

IDF_FILE      = os.path.join("..", "..", "DOAS_wNeutralSupplyAir_wFanCoilUnits.idf")
WEATHER_FILE  = os.path.join("..", "..", "FIN_TR_Tampere.Satakunnankatu.027440_TMYx.2004-2018.epw")
MODEL_PATH    = os.path.join("models", "sac_bc_hvac")   # path to saved model (without .zip)
EVAL_OUT      = "vesa_3_output/eval"


# =========================================================================
# Evaluation
# =========================================================================

def evaluate(model_path, config, csv_prefix="eval"):
    env = Monitor(EnergyPlusEnv(config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAC.load(model_path, env=env, device=device)

    obs_names = config["observations"]
    act_names = config["rl_actions"]

    obs, _ = env.reset()
    done, truncated = False, False
    rows = []

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, _ = env.step(action)

        row = {name: float(obs[i]) for i, name in enumerate(obs_names)}
        row.update({name: float(action[i]) for i, name in enumerate(act_names)})
        row["reward"] = float(reward)
        row["done"] = done
        row["truncated"] = truncated
        rows.append(row)
        obs = next_obs

    env.close()

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = f"{csv_prefix}_{ts}.csv"
    df.to_csv(path, index=False)
    print(f"Saved evaluation ({len(df)} steps): {path}")
    return df


if __name__ == "__main__":
    eval_config = _build_config(IDF_FILE, WEATHER_FILE, EVAL_OUT, "test")
    evaluate(MODEL_PATH, eval_config, csv_prefix="sac_eval")