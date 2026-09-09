#!/usr/bin/env python3
"""
Step 6 of experiments/kan-identifiability: is the end-to-end curve agreement
(RAW=QUOTIENTED=0.0105 in kan_seed_variance.json) non-trivial, or does it just
restate "all seeds fit exp(-u) well"?

Distinguishes, on the same 10 seeds / same common grid used throughout:
  (a) seed-to-seed spread:   mean_{i<j} ||f_i - f_j||_2 / ||f_i||_2
  (b) deviation from truth:  mean_i     ||f_i - y_true||_2 / ||y_true||_2

U_SCALE CARE: the model's input domain is NORMALIZED, x = u/U_SCALE in
[0.012, 1.0] (matches training). The comparison target is exp(-u) in the
PHYSICAL domain, i.e. y_true = exp(-x*U_SCALE), NOT exp(-x).

If (a) << (b): seeds agree with each other more tightly than any agrees with
the truth -- a real statement about the estimator (e.g. a shared, non-truth
bias), not just "training converges." If (a) ~ (b): "identified" adds nothing
beyond "they all converged" -- (a)'s smallness is explained entirely by (b)'s
smallness.

Reuses train_one_seed / extract_full_model / summarize from
run_kan_seed_variance.py -- same code, not reimplemented.

Usage:
    python experiments/identifiability_gate/run_kan_endtoend_stability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_kan_seed_variance import (  # noqa: E402
    train_one_seed, extract_full_model, summarize, SEEDS, U_SCALE, KAN_HIDDEN,
    N_EDGE_PTS, RESULTS_DIR,
)

X_MIN, X_MAX = 0.012, 1.0   # normalized input domain, matches Task A / Step 2-3


def main() -> None:
    print(f"Training {len(SEEDS)} seeds (hidden={KAN_HIDDEN}) for end-to-end stability decomposition...")
    runs = [train_one_seed(s, hidden=KAN_HIDDEN) for s in SEEDS]
    for r in runs:
        print(f"  seed={r['seed']:2d}  test_r2={r['test_r2']:.6f}")

    curves = [extract_full_model(r["model"], X_MIN, X_MAX, n_pts=N_EDGE_PTS) for r in runs]

    x_vals = np.linspace(X_MIN, X_MAX, N_EDGE_PTS)
    u_phys = x_vals * U_SCALE                      # undo input normalization
    y_true = np.exp(-u_phys)                        # true target in physical domain

    n = len(SEEDS)
    pair_idx = [(i, j) for i in range(n) for j in range(n) if i < j]

    seed_to_seed = [float(np.linalg.norm(curves[i] - curves[j]) / (np.linalg.norm(curves[i]) + 1e-12))
                     for i, j in pair_idx]
    deviation_from_truth = [float(np.linalg.norm(curves[i] - y_true) / (np.linalg.norm(y_true) + 1e-12))
                              for i in range(n)]

    a_summary = summarize(seed_to_seed)
    b_summary = summarize(deviation_from_truth)

    print(f"\n(a) seed-to-seed spread:  mean={a_summary['mean']:.5f}  std={a_summary['std']:.5f}  "
          f"(n_pairs={len(pair_idx)})")
    print(f"(b) deviation from truth: mean={b_summary['mean']:.5f}  std={b_summary['std']:.5f}  "
          f"(n_seeds={n})")

    ratio = a_summary["mean"] / (b_summary["mean"] + 1e-12)
    print(f"\nratio (a)/(b) = {ratio:.3f}")

    if ratio < 0.5:
        verdict = "A_MUCH_LESS_THAN_B"
        verdict_text = (f"(a)={a_summary['mean']:.5f} << (b)={b_summary['mean']:.5f} (ratio={ratio:.3f}): "
                         f"seeds agree with EACH OTHER more tightly than any individual seed agrees "
                         f"with the true exp(-u). This is a real, non-trivial statement about the "
                         f"estimator -- the trained models converge to a highly consistent function "
                         f"that is not simply 'close to truth with independent noise'; there is a "
                         f"shared systematic component (bias/approximation error) that is itself "
                         f"stable across random seeds.")
    elif ratio > 0.8:
        verdict = "A_ROUGHLY_EQUALS_B"
        verdict_text = (f"(a)={a_summary['mean']:.5f} ~ (b)={b_summary['mean']:.5f} (ratio={ratio:.3f}): "
                         f"seed-to-seed agreement is not meaningfully tighter than each seed's "
                         f"agreement with the truth. 'Identified' adds nothing here beyond 'they all "
                         f"converged to a good fit' -- (a)'s smallness is fully explained by (b)'s "
                         f"smallness, not by any additional stability of the estimator itself.")
    else:
        verdict = "INTERMEDIATE"
        verdict_text = (f"(a)={a_summary['mean']:.5f} vs (b)={b_summary['mean']:.5f} (ratio={ratio:.3f}): "
                         f"intermediate -- seeds agree with each other somewhat more tightly than with "
                         f"the truth, but not by an order of magnitude. Partial evidence of a "
                         f"non-trivial shared estimator behavior beyond 'they all converged'.")

    print(f"\nVERDICT: {verdict}")
    print(f"  {verdict_text}")

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    out = dict(
        hidden=KAN_HIDDEN, seeds=SEEDS, n_grid_pts=N_EDGE_PTS,
        x_domain=[X_MIN, X_MAX], u_scale=U_SCALE,
        seed_to_seed_spread=a_summary,
        deviation_from_truth=b_summary,
        ratio_a_over_b=ratio,
        verdict=verdict,
        verdict_text=verdict_text,
    )
    with open(RESULTS_DIR / "kan_endtoend_stability.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'kan_endtoend_stability.json'}")


if __name__ == "__main__":
    main()
