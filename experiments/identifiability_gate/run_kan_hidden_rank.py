#!/usr/bin/env python3
"""
Step 5 of experiments/kan-identifiability: is the hidden-unit decomposition
nominal? For one trained model per width, evaluate layer1's output on the
common input grid to form H (n_grid x hidden), and report H's singular value
spectrum (normalized by the largest) and effective rank at thresholds 1e-2,
1e-3.

PURPOSE: if H is effectively rank-1 at hidden=4, the "4-way decomposition" is
nominal -- the model is really a 1->1 map through a redundant parameterization,
and the paper should say that rather than implying a genuine multi-branch
decomposition.

Uses seed=0 (first seed in the fixed SEEDS list) for one representative model
per width. Reuses train_one_seed from run_kan_seed_variance.py.

Usage:
    python experiments/identifiability_gate/run_kan_hidden_rank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_kan_seed_variance import train_one_seed, N_EDGE_PTS, RESULTS_DIR  # noqa: E402

WIDTHS = [1, 2, 4, 8]
REPRESENTATIVE_SEED = 0
X_MIN, X_MAX = 0.012, 1.0


def main() -> None:
    results = {}
    for h in WIDTHS:
        r = train_one_seed(REPRESENTATIVE_SEED, hidden=h)
        model = r["model"]
        x_vals = torch.linspace(X_MIN, X_MAX, N_EDGE_PTS).unsqueeze(1)
        with torch.no_grad():
            H = model.layer1(x_vals).numpy()  # [n_grid, hidden]

        s = np.linalg.svd(H, compute_uv=False)
        s_norm = (s / s[0]).tolist()
        eff_rank_1e2 = int(np.sum(np.array(s_norm) > 1e-2))
        eff_rank_1e3 = int(np.sum(np.array(s_norm) > 1e-3))

        results[h] = dict(
            hidden=h, test_r2=r["test_r2"],
            singular_values_normalized=s_norm,
            effective_rank_1e2=eff_rank_1e2,
            effective_rank_1e3=eff_rank_1e3,
        )
        print(f"hidden={h}  test_r2={r['test_r2']:.6f}")
        print(f"  normalized singular values: {[f'{v:.4f}' for v in s_norm]}")
        print(f"  effective rank (>1e-2): {eff_rank_1e2}   effective rank (>1e-3): {eff_rank_1e3}")

    print(f"\n{'='*70}\nSTEP 5 READING\n{'='*70}")
    h4 = results.get(4)
    if h4 is not None:
        if h4["effective_rank_1e2"] == 1:
            reading = ("H is effectively RANK-1 at hidden=4 (threshold 1e-2): the 4-way "
                       "decomposition is NOMINAL. The model is really a 1->1 map computed "
                       "through 4 redundant, near-collinear branches -- the paper should not "
                       "imply a genuine multi-branch decomposition.")
        elif h4["effective_rank_1e2"] >= 4:
            reading = ("H is full rank at hidden=4 (threshold 1e-2): all 4 hidden units "
                       "contribute independent directions. The 4-way decomposition is genuine "
                       "in the sense that no single unit's activation pattern is redundant "
                       "with another's -- edge instability across seeds is not explained by "
                       "trivial rank-collapse.")
        else:
            reading = (f"H has effective rank {h4['effective_rank_1e2']} (of 4 possible) at "
                       f"hidden=4 (threshold 1e-2): partial redundancy -- more than 1 but fewer "
                       f"than 4 independent directions are actually used.")
        print(reading)
    else:
        reading = "hidden=4 not in WIDTHS; no reading computed."

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    out = dict(widths=WIDTHS, representative_seed=REPRESENTATIVE_SEED,
               results=results, reading_hidden4=reading)
    with open(RESULTS_DIR / "kan_hidden_rank.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'kan_hidden_rank.json'}")


if __name__ == "__main__":
    main()
