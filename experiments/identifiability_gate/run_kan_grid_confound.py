#!/usr/bin/env python3
"""
Step 11 follow-up: is the grid -> h-vs-affine trend REAL, or is it a fit-quality
confound?

run_kan_grid_sweep.py found h-vs-affine rising monotonically with grid
(0.0357 -> 0.0830 -> 0.1261 at grid 6/12/24, nothing usable at 48). But
convergence degrades systematically over exactly the same axis (mean test
R^2 0.9979 -> 0.9858 -> 0.9478 -> 0.8625), and the well-converged pair counts
are 1, 3, 6, 0. A worse-fit model has more slack for its reparameterization to
look non-affine, so the two explanations are perfectly confounded in that table.

This disentangles them: run the gauge test on ALL pairs at every grid (not just
well-converged ones), record each pair's fit quality alongside its h-vs-affine
residual, then
  (a) correlate h-vs-affine with fit quality directly, and
  (b) compare grids WITHIN a matched fit-quality band, so grid is the only
      thing that differs.
If the trend survives at matched fit quality, discretization is supported. If it
vanishes, the Step 11 verdict was the confound and is reported as such.

Usage:
    python experiments/identifiability_gate/run_kan_grid_confound.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_kan_seed_variance import train_one_seed, summarize, SEEDS, N_EDGE_PTS, RESULTS_DIR  # noqa: E402
from kan_composition_gauge_lib import (  # noqa: E402
    composition_gauge_test, make_marginal_layer_fn, make_layer_fn,
)

GRIDS = [6, 12, 24, 48]
HIDDEN = 1
X_MIN, X_MAX = 0.012, 1.0
BAND = (0.99, 1.0)   # matched fit-quality band, chosen to be populated at every grid


def main() -> None:
    print(f"Step 11 confound check: gauge test on ALL pairs, grids {GRIDS}\n")
    all_rows = []

    for grid in GRIDS:
        runs = [train_one_seed(s, hidden=HIDDEN, grid=grid) for s in SEEDS]
        r2 = {r["seed"]: r["test_r2"] for r in runs}
        models = {r["seed"]: r["model"] for r in runs}
        x_grid = np.linspace(X_MIN, X_MAX, N_EDGE_PTS)
        n_usable = 0
        for i, j in itertools.combinations(SEEDS, 2):
            phi1_i = make_marginal_layer_fn(models[i].layer1, in_features=1, input_idx=0)
            phi1_j = make_marginal_layer_fn(models[j].layer1, in_features=1, input_idx=0)
            g = composition_gauge_test(x_grid, phi1_i(x_grid), phi1_j,
                                        make_layer_fn(models[i].layer2),
                                        make_layer_fn(models[j].layer2))
            if not g["usable"]:
                continue
            n_usable += 1
            all_rows.append(dict(grid=grid, i=i, j=j,
                                  h_affine_residual=g["h_affine_residual"],
                                  composition_residual=g["composition_residual_raw"],
                                  min_r2=min(r2[i], r2[j]), mean_r2=(r2[i] + r2[j]) / 2))
        print(f"  grid={grid:2d}: {n_usable}/45 pairs usable, "
              f"seed R^2 range [{min(r2.values()):.4f}, {max(r2.values()):.4f}]")

    arr_grid = np.array([r["grid"] for r in all_rows], dtype=float)
    arr_h = np.array([r["h_affine_residual"] for r in all_rows])
    arr_r2 = np.array([r["min_r2"] for r in all_rows])

    r_grid = float(np.corrcoef(np.log2(arr_grid), arr_h)[0, 1])
    r_fit = float(np.corrcoef(arr_r2, arr_h)[0, 1])
    print(f"\n=== (a) direct correlations over all {len(all_rows)} usable pairs ===")
    print(f"  corr(log2 grid, h-vs-affine) = {r_grid:+.4f}")
    print(f"  corr(min pair R^2, h-vs-affine) = {r_fit:+.4f}   (negative = worse fit -> larger h)")

    print(f"\n=== (b) matched fit-quality band: min pair R^2 in [{BAND[0]}, {BAND[1]}] ===")
    hdr = f"  {'grid':>5s} {'n_pairs':>8s} {'h_mean':>9s} {'h_std':>9s} {'min_r2_mean':>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    banded = {}
    for grid in GRIDS:
        sel = [r for r in all_rows if r["grid"] == grid and BAND[0] <= r["min_r2"] <= BAND[1]]
        if sel:
            hs = [r["h_affine_residual"] for r in sel]
            banded[grid] = dict(n=len(sel), h_mean=float(np.mean(hs)), h_std=float(np.std(hs)),
                                 r2_mean=float(np.mean([r["min_r2"] for r in sel])))
            print(f"  {grid:5d} {len(sel):8d} {banded[grid]['h_mean']:9.4f} "
                  f"{banded[grid]['h_std']:9.4f} {banded[grid]['r2_mean']:12.5f}")
        else:
            banded[grid] = None
            print(f"  {grid:5d} {0:8d} {'--':>9s} {'--':>9s} {'--':>12s}")

    present = [(g, banded[g]["h_mean"]) for g in GRIDS if banded[g]]
    band_monotonic = all(present[k][1] <= present[k + 1][1] + 1e-12 for k in range(len(present) - 1))
    print(f"\n  monotonic within the matched band: {band_monotonic}  "
          f"({len(present)} grids populated)")

    if band_monotonic and len(present) >= 3:
        verdict = "GRID_EFFECT_SURVIVES_MATCHING"
        text = (f"h-vs-affine still rises with grid at matched fit quality "
                f"({', '.join(f'grid{g}={h:.4f}' for g, h in present)}), so the trend is not merely "
                f"a fit-quality artifact. Discretization genuinely restricts which "
                f"reparameterizations are realizable, supporting the Step 11 verdict.")
    elif len(present) < 3:
        verdict = "UNDERPOWERED"
        text = (f"Only {len(present)} grid(s) have pairs inside the matched fit-quality band, so the "
                f"grid effect cannot be separated from the fit-quality confound with this data. "
                f"The Step 11 monotonic trend stands as SUGGESTIVE ONLY: it is perfectly confounded "
                f"with the systematic degradation of convergence as grid grows "
                f"(corr(min R^2, h) = {r_fit:+.4f}), and rests on 1, 3 and 6 pairs at grids 6, 12, 24 "
                f"with nothing usable at 48.")
    else:
        verdict = "GRID_EFFECT_IS_CONFOUND"
        text = (f"Once fit quality is matched, h-vs-affine no longer rises monotonically with grid "
                f"({', '.join(f'grid{g}={h:.4f}' for g, h in present)}). The Step 11 trend is "
                f"explained by convergence degrading with grid, not by discretization limiting the "
                f"realizable gauge. The discretization explanation is NOT supported.")

    print(f"\nVERDICT: {verdict}\n  {text}")

    out = dict(grids=GRIDS, band=BAND, n_pairs=len(all_rows), rows=all_rows,
               corr_log2grid_h=r_grid, corr_minr2_h=r_fit,
               banded=banded, band_monotonic=band_monotonic,
               verdict=verdict, verdict_text=text)
    with open(RESULTS_DIR / "kan_grid_confound.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'kan_grid_confound.json'}")


if __name__ == "__main__":
    main()
