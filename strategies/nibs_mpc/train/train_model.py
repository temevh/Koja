#!/usr/bin/env python3
"""Train ensemble of MLP dynamics models.

Trains 5 models with different random seeds on the prepared dataset.
Each model uses AdamW + cosine LR decay + early stopping.

Usage:
    cd strategies/nibs_mpc/
    python train/train_model.py
    python train/train_model.py --n-ensemble 3 --epochs 200  # quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dynamics_model import DynamicsModel, TEMP_DIM, CO2_DIM, ENERGY_DIM

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_one_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    seed: int,
    epochs: int = 300,
    batch_size: int = 2048,
    lr: float = 1e-3,
    patience: int = 15,
    device: str = "cpu",
) -> tuple[DynamicsModel, dict]:
    """Train a single dynamics model. Returns (model, training_stats)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DynamicsModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
    )
    use_cuda = str(device).startswith("cuda")
    loader_kwargs = {"pin_memory": use_cuda, "num_workers": 2 if use_cuda else 0,
                     "persistent_workers": use_cuda}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, **loader_kwargs)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_temp_rmse": [], "val_co2_rmse": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred_temp, pred_co2, pred_energy = model(xb)
            pred = torch.cat([pred_temp, pred_co2, pred_energy], dim=-1)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        val_temp_sq, val_co2_sq = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred_temp, pred_co2, pred_energy = model(xb)
                pred = torch.cat([pred_temp, pred_co2, pred_energy], dim=-1)
                val_losses.append(nn.functional.mse_loss(pred, yb).item())
                val_temp_sq.append(((pred_temp - yb[:, :TEMP_DIM]) ** 2).mean().item())
                val_co2_sq.append(((pred_co2 - yb[:, TEMP_DIM:TEMP_DIM + CO2_DIM]) ** 2).mean().item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_temp_rmse = np.sqrt(np.mean(val_temp_sq))
        val_co2_rmse = np.sqrt(np.mean(val_co2_sq))

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_temp_rmse"].append(float(val_temp_rmse))
        history["val_co2_rmse"].append(float(val_co2_rmse))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"temp_rmse={val_temp_rmse:.6f}  co2_rmse={val_co2_rmse:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if wait >= patience:
            print(f"  Early stop at epoch {epoch+1} (patience={patience})")
            break

    model.load_state_dict(best_state)
    model.eval()

    stats = {
        "seed": seed,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "total_epochs": len(history["train_loss"]),
    }
    return model, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MPC dynamics ensemble")
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' uses CUDA if available, or 'cpu'/'cuda'")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}", end="")
    if args.device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    # Load dataset
    data = np.load(MODELS_DIR / "dataset.npz")
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val, Y_val = data["X_val"], data["Y_val"]
    print(f"Dataset: {len(X_train)} train, {len(X_val)} val")
    print(f"Input dim: {X_train.shape[1]}, Output dim: {Y_train.shape[1]}")
    print(f"Batch size: {args.batch_size}\n")

    all_stats = []
    t0 = time.time()

    for i in range(args.n_ensemble):
        seed = 100 + i * 17
        print(f"Training ensemble member {i+1}/{args.n_ensemble} (seed={seed})")

        model, stats = train_one_model(
            X_train, Y_train, X_val, Y_val,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device=args.device,
        )

        save_path = MODELS_DIR / f"ensemble_{i}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"  Saved → {save_path} (best val loss: {stats['best_val_loss']:.6f}, "
              f"epoch {stats['best_epoch']})\n")
        all_stats.append(stats)

    elapsed = time.time() - t0
    print(f"Done: {args.n_ensemble} models in {elapsed:.1f}s")

    # Summary
    val_losses = [s["best_val_loss"] for s in all_stats]
    print(f"Val loss: mean={np.mean(val_losses):.6f}, "
          f"min={np.min(val_losses):.6f}, max={np.max(val_losses):.6f}")


if __name__ == "__main__":
    main()
