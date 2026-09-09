#!/usr/bin/env python3
"""
Deterministic reproduction of Task A (u -> exp(-u)) from run_kan_interpretability.py
(commit a2d4be6, "KAN interpretability -- spline extraction vs MRI signal model").

WHY THIS SCRIPT EXISTS: that experiment's SmallKAN model was trained in-memory and
never saved to disk, so experiments/kan-identifiability's Step 1 gauge check has no
existing checkpoint to load. This script re-runs the exact same architecture,
hyperparameters, and seed to regenerate an equivalent trained model and checkpoints
it, so the gauge check itself (run_kan_gauge_check.py) can load-only, no training.

Deliberately forced to CPU (the original preferred MPS when available) so this
reproduction is bit-for-bit deterministic run to run; MPS training is not
guaranteed deterministic. Architecture, seed (42), data, hyperparameters, and
training procedure are otherwise identical to the original.

Usage:
    python experiments/identifiability_gate/reproduce_kan_task_a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mhwf_pikan.core.fno_kan import KANLinear  # noqa: E402  (tracked, has spline_scale)

SEED = 42
N_TRAIN = 8_000
N_TEST = 2_000
TRAIN_EPOCHS = 600
KAN_LR = 3e-3
KAN_GRID = 12
KAN_K = 3
KAN_HIDDEN = 4
U_SCALE = 4.0

CHECKPOINT_PATH = _REPO_ROOT / "checkpoints" / "kan_task_a_repro.pt"


class SmallKAN(nn.Module):
    """2-layer KAN built from the repo KANLinear: [in -> H -> out]. Verbatim from a2d4be6."""

    def __init__(self, in_dim: int, hidden: int = KAN_HIDDEN,
                 out_dim: int = 1, grid: int = KAN_GRID, k: int = KAN_K) -> None:
        super().__init__()
        self.layer1 = KANLinear(in_dim, hidden, grid_size=grid, spline_order=k)
        self.layer2 = KANLinear(hidden, out_dim, grid_size=grid, spline_order=k)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_model(model: nn.Module, X_tr: torch.Tensor, y_tr: torch.Tensor,
                 n_epochs: int, lr: float, label: str = "") -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
    log_ev = max(1, n_epochs // 4)
    losses = []
    for ep in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.mse_loss(model(X_tr).squeeze(-1), y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if ep % log_ev == 0:
            print(f"    [{label}] ep {ep:4d}/{n_epochs}  mse={loss.item():.6f}")
    return losses


def eval_mse(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X).squeeze(-1)
    return float(F.mse_loss(pred, y).item())


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    print(f"Device: {device} (forced, for determinism)")
    print(f"KAN: [1,{KAN_HIDDEN},1]  grid={KAN_GRID}  k={KAN_K}  "
          f"epochs={TRAIN_EPOCHS}  N_train={N_TRAIN}  seed={SEED}")

    rng = np.random.default_rng(SEED)
    u_tr = rng.uniform(0.05, U_SCALE, N_TRAIN).astype(np.float32)
    u_te = rng.uniform(0.05, U_SCALE, N_TEST).astype(np.float32)
    y_tr = np.exp(-u_tr)
    y_te = np.exp(-u_te)

    X_tr = torch.from_numpy(u_tr / U_SCALE).unsqueeze(1)
    X_te = torch.from_numpy(u_te / U_SCALE).unsqueeze(1)
    Y_tr = torch.from_numpy(y_tr)
    Y_te = torch.from_numpy(y_te)

    kan = SmallKAN(in_dim=1).to(device)
    X_tr_d, Y_tr_d, X_te_d, Y_te_d = (t.to(device) for t in (X_tr, Y_tr, X_te, Y_te))
    train_model(kan, X_tr_d, Y_tr_d, TRAIN_EPOCHS, KAN_LR, "KAN-A-repro")
    kan_mse = eval_mse(kan, X_te_d, Y_te_d)

    print(f"\nFinal test MSE: {kan_mse:.3e}  (a2d4be6 reported Pearson r=0.9997 vs true exp(-u))")

    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": kan.state_dict(),
        "config": dict(in_dim=1, hidden=KAN_HIDDEN, out_dim=1, grid=KAN_GRID, k=KAN_K),
        "task_mse": kan_mse,
        "source_commit": "a2d4be6",
        "source_script": "experiments/identifiability_gate/run_kan_interpretability.py",
        "task": "A: u -> exp(-u)",
        "seed": SEED,
        "device_trained_on": str(device),
        "note": "Deterministic reproduction for experiments/kan-identifiability Step 1; "
                "original model was never checkpointed.",
    }, CHECKPOINT_PATH)
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
