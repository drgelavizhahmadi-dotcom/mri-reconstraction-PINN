#!/usr/bin/env python3
"""
Step 4 of experiments/kan-identifiability: does edge instability track hidden-unit
REDUNDANCY, or is it present even with no redundancy to trade?

Reruns the exact Step 2/3 protocol (run_kan_seed_variance.run_seed_variance --
same metrics code, not reimplemented) at KAN_HIDDEN in {1, 2, 4, 8}, same 10
seeds (0-9), everything else identical (grid=12, k=3, epochs=600, LR schedule,
grad clipping, data distribution).

PREDICTION (redundancy hypothesis): edge instability should DECREASE
monotonically as width decreases; at hidden=1 there is nothing to redistribute
mass across, so edges should be stable modulo affine (QUOTIENTED small, fitted
a clustered away from zero). Reported as measured -- not tuned toward.

Usage:
    python experiments/identifiability_gate/run_kan_width_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_kan_seed_variance import run_seed_variance, SEEDS, RESULTS_DIR  # noqa: E402

WIDTHS = [1, 2, 4, 8]
QUOT_SMALL_THRESHOLD = 0.10
CONVERGENCE_R2_THRESHOLD = 0.999


def main() -> None:
    print(f"Width sweep: hidden in {WIDTHS}, seeds={SEEDS} (fixed across all widths)\n")

    results = {}
    for h in WIDTHS:
        print(f"\n{'='*70}\nHIDDEN = {h}\n{'='*70}")
        out = run_seed_variance(hidden=h, seeds=SEEDS, save=True, verbose=True)
        results[h] = out

    print(f"\n\n{'='*70}\nWIDTH SWEEP SUMMARY\n{'='*70}")
    header = (f"{'hidden':>6} | {'test_R2_min':>11} | {'test_MSE_mean':>13} | "
              f"{'RAW_mean':>9} | {'QUOT_mean':>9} | {'a_mean':>8} | {'a_std':>8} | "
              f"{'converged':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for h in WIDTHS:
        out = results[h]
        r2_min = out["convergence"]["test_r2"]["min"]
        mse_mean = out["convergence"]["test_mse"]["mean"]
        raw_mean = out["overall_raw"]["mean"]
        quot_mean = out["overall_quotiented"]["mean"]
        a_mean = out["layer1_aggregate"]["a"]["mean"]
        a_std = out["layer1_aggregate"]["a"]["std"]
        converged = r2_min > CONVERGENCE_R2_THRESHOLD
        rows.append(dict(hidden=h, test_r2_min=r2_min, test_mse_mean=mse_mean,
                          raw_mean=raw_mean, quot_mean=quot_mean,
                          a_mean=a_mean, a_std=a_std, converged=converged))
        print(f"{h:>6} | {r2_min:>11.6f} | {mse_mean:>13.3e} | {raw_mean:>9.3f} | "
              f"{quot_mean:>9.3f} | {a_mean:>8.3f} | {a_std:>8.3f} | {str(converged):>9}")

    not_converged = [r["hidden"] for r in rows if not r["converged"]]
    if not_converged:
        print(f"\nCONVERGENCE WARNING: hidden={not_converged} did NOT reach "
              f"test R^2 > {CONVERGENCE_R2_THRESHOLD}. Its edge-stability numbers "
              f"above are NOT comparable to the other widths -- a poorly-converged "
              f"model's edges can look 'unstable' simply because the fit itself "
              f"hasn't settled, not because of any redundancy/identifiability effect.")
    else:
        print(f"\nAll {len(WIDTHS)} widths converged (test R^2 > {CONVERGENCE_R2_THRESHOLD}); "
              f"edge-stability numbers are comparable across the sweep.")

    quot_by_width = [r["quot_mean"] for r in rows]  # ordered by WIDTHS = [1,2,4,8]
    is_monotonic_increasing_with_width = all(
        quot_by_width[i] <= quot_by_width[i + 1] + 1e-9 for i in range(len(quot_by_width) - 1)
    )
    hidden1 = rows[0]
    hidden1_stable = hidden1["quot_mean"] < QUOT_SMALL_THRESHOLD
    hidden1_a_away_from_zero = abs(hidden1["a_mean"]) > 3 * hidden1["a_std"] if hidden1["a_std"] > 0 else abs(hidden1["a_mean"]) > 0.5
    prediction_holds = is_monotonic_increasing_with_width and hidden1_stable and hidden1_a_away_from_zero

    print(f"\n{'='*70}\nPREDICTION CHECK (redundancy hypothesis)\n{'='*70}")
    print(f"  QUOTIENTED monotonically non-decreasing with width "
          f"({dict(zip(WIDTHS, [f'{q:.3f}' for q in quot_by_width]))}): "
          f"{is_monotonic_increasing_with_width}")
    print(f"  hidden=1 QUOTIENTED small (<{QUOT_SMALL_THRESHOLD}): {hidden1['quot_mean']:.3f} -> "
          f"{hidden1_stable}")
    print(f"  hidden=1 fitted a clustered away from zero "
          f"(mean={hidden1['a_mean']:.3f}, std={hidden1['a_std']:.3f}): {hidden1_a_away_from_zero}")

    if prediction_holds:
        prediction_verdict = ("PREDICTION HOLDS: edge instability decreases as width decreases, "
                               "and hidden=1 edges are stable modulo affine. Redundancy across "
                               "hidden units is a viable explanation for the Step 3 instability "
                               "at hidden=4.")
    else:
        reasons = []
        if not is_monotonic_increasing_with_width:
            reasons.append("QUOTIENTED does not decrease monotonically with width")
        if not hidden1_stable:
            reasons.append(f"hidden=1 QUOTIENTED is still large ({hidden1['quot_mean']:.3f})")
        if not hidden1_a_away_from_zero:
            reasons.append(f"hidden=1 fitted a is still centered near zero "
                            f"(mean={hidden1['a_mean']:.3f}, std={hidden1['a_std']:.3f})")
        prediction_verdict = ("PREDICTION FAILS: " + "; ".join(reasons) + ". "
                               "Redundancy across hidden units is NOT the (sole) mechanism -- "
                               "even hidden=1, with no branches to trade mass between, shows "
                               "edge instability. Something else (e.g. genuinely different "
                               "spline/base-path solutions fitting the same function equally "
                               "well within one edge) must be contributing.")

    print(f"\n{prediction_verdict}")

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    summary = dict(
        widths=WIDTHS, seeds=SEEDS,
        rows=rows,
        prediction_holds=prediction_holds,
        prediction_verdict=prediction_verdict,
        not_converged_widths=not_converged,
    )
    with open(RESULTS_DIR / "kan_width_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'kan_width_sweep_summary.json'}")
    print(f"  → per-width detail: kan_seed_variance.json (hidden=4), "
          f"kan_seed_variance_hidden{{1,2,8}}.json")


if __name__ == "__main__":
    main()
