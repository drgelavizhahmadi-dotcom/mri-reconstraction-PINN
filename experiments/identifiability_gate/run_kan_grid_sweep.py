#!/usr/bin/env python3
"""
Step 11 of experiments/kan-identifiability: WHY did h come out near-affine at
d=1, when the theoretical d=1 gauge group is the full diffeomorphism group?

HYPOTHESIS: the finite B-spline grid restricts which reparameterizations
layer2 can actually represent. At grid=12 only near-affine h survives; a
richer grid should let training realize wilder h. So h's distance from affine
should GROW monotonically with grid size.

Protocol: d=1 (hidden=1), grid in {6, 12, 24, 48}, k=3, 10 seeds (same fixed
list), same target (u -> exp(-u)), same schedule/epochs/clipping, forced-CPU
determinism. Per grid: h-vs-affine residual (Step 8 method, well-converged
pairs only), layer1/layer2 QUOTIENTED, distribution of a, per-seed convergence.

If the residual does NOT increase monotonically with grid, the discretization
explanation is wrong and is reported as wrong.

Reuses train_one_seed / raw_and_quotiented / summarize and
composition_gauge_test -- no metric reimplementation.

Usage:
    python experiments/identifiability_gate/run_kan_grid_sweep.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_kan_seed_variance import (  # noqa: E402
    train_one_seed, extract_layer1_edge, extract_layer2_edge,
    raw_and_quotiented, summarize, SEEDS, N_EDGE_PTS, RESULTS_DIR,
)
from kan_composition_gauge_lib import (  # noqa: E402
    composition_gauge_test, make_marginal_layer_fn, make_layer_fn,
)

GRIDS = [6, 12, 24, 48]
HIDDEN = 1
R2_THRESHOLD = 0.999
X_MIN, X_MAX = 0.012, 1.0
U_MIN, U_MAX = -1.0, 1.0


def main() -> None:
    print(f"Step 11 grid sweep: grid in {GRIDS}, hidden={HIDDEN}, seeds={SEEDS}")
    per_grid = {}

    for grid in GRIDS:
        print(f"\n{'='*72}\nGRID = {grid}\n{'='*72}")
        runs = []
        for s in SEEDS:
            r = train_one_seed(s, hidden=HIDDEN, grid=grid)
            runs.append(r)
            print(f"  seed={s:2d}  final_train_loss={r['final_train_loss']:.3e}  "
                  f"test_mse={r['test_mse']:.3e}  test_r2={r['test_r2']:.6f}")

        r2s = [r["test_r2"] for r in runs]
        r2_summary = summarize(r2s)
        wc = sorted(r["seed"] for r in runs if r["test_r2"] > R2_THRESHOLD)
        print(f"  convergence: R^2 mean={r2_summary['mean']:.6f} min={r2_summary['min']:.6f}  "
              f"well-converged {len(wc)}/{len(SEEDS)}: {wc}")

        # edge comparison (all pairs and filtered), same metrics as every prior step
        l1 = [extract_layer1_edge(r["model"], 0, X_MIN, X_MAX) for r in runs]
        l2 = [extract_layer2_edge(r["model"], 0, U_MIN, U_MAX, hidden=HIDDEN) for r in runs]
        pair_idx = [(i, j) for i in range(len(SEEDS)) for j in range(len(SEEDS)) if i < j]
        keep = [k for k, p in enumerate(pair_idx) if p[0] in wc and p[1] in wc]

        def cmp(curves):
            pairs = [raw_and_quotiented(curves[i], curves[j]) for i, j in pair_idx]
            filt = [pairs[k] for k in keep]
            return dict(
                all_quotiented=summarize([p["quotiented"] for p in pairs]),
                all_a=summarize([p["a"] for p in pairs]),
                filtered_quotiented=summarize([p["quotiented"] for p in filt]) if filt else None,
                filtered_a=summarize([p["a"] for p in filt]) if filt else None,
                n_filtered=len(filt))

        l1_res, l2_res = cmp(l1), cmp(l2)
        print(f"  layer1 QUOTIENTED: all={l1_res['all_quotiented']['mean']:.4f}  "
              f"filtered={l1_res['filtered_quotiented']['mean'] if l1_res['filtered_quotiented'] else float('nan'):.4f}")
        print(f"  layer2 QUOTIENTED: all={l2_res['all_quotiented']['mean']:.4f}  "
              f"filtered={l2_res['filtered_quotiented']['mean'] if l2_res['filtered_quotiented'] else float('nan'):.4f}")

        # composition gauge on well-converged pairs
        x_grid = np.linspace(X_MIN, X_MAX, N_EDGE_PTS)
        models = {r["seed"]: r["model"] for r in runs}
        gauge_rows = []
        for i, j in itertools.combinations(wc, 2):
            phi1_i = make_marginal_layer_fn(models[i].layer1, in_features=1, input_idx=0)
            phi1_j = make_marginal_layer_fn(models[j].layer1, in_features=1, input_idx=0)
            phi2_i = make_layer_fn(models[i].layer2)
            phi2_j = make_layer_fn(models[j].layer2)
            g = composition_gauge_test(x_grid, phi1_i(x_grid), phi1_j, phi2_i, phi2_j)
            if g["usable"]:
                gauge_rows.append(dict(i=i, j=j, h_affine_residual=g["h_affine_residual"],
                                        h_affine_a=g["h_affine_a"],
                                        composition_residual_raw=g["composition_residual_raw"],
                                        interval_frac=g["interval_frac"]))
            else:
                gauge_rows.append(dict(i=i, j=j, usable=False,
                                        interval_frac=g.get("interval_frac", 0.0)))

        usable = [g for g in gauge_rows if "h_affine_residual" in g]
        h_res = summarize([g["h_affine_residual"] for g in usable]) if usable else None
        comp_res = summarize([g["composition_residual_raw"] for g in usable]) if usable else None
        print(f"  composition gauge: {len(usable)}/{len(gauge_rows)} usable pairs  "
              f"h-vs-affine mean={h_res['mean']:.4f}" if usable else "  no usable pairs")

        per_grid[grid] = dict(
            grid=grid, convergence=dict(test_r2=r2_summary, well_converged=wc,
                                         n_well_converged=len(wc)),
            per_seed=[dict(seed=r["seed"], final_train_loss=r["final_train_loss"],
                            test_mse=r["test_mse"], test_r2=r["test_r2"]) for r in runs],
            layer1=l1_res, layer2=l2_res,
            gauge_pairs=gauge_rows,
            h_affine_residual=h_res, composition_residual=comp_res,
            n_usable_gauge_pairs=len(usable))

    # ── monotonicity check ──────────────────────────────────────────────────
    print(f"\n{'='*72}\nSTEP 11 SUMMARY\n{'='*72}")
    hdr = (f"  {'grid':>5s} {'n_wc':>5s} {'n_pairs':>8s} {'h_vs_affine':>12s} {'comp_resid':>11s} "
           f"{'L1_QUOT_f':>10s} {'L2_QUOT_f':>10s} {'a_mean_f':>9s} {'a_std_f':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    h_series = []
    for grid in GRIDS:
        g = per_grid[grid]
        h_mean = g["h_affine_residual"]["mean"] if g["h_affine_residual"] else float("nan")
        c_mean = g["composition_residual"]["mean"] if g["composition_residual"] else float("nan")
        l1f = g["layer1"]["filtered_quotiented"]["mean"] if g["layer1"]["filtered_quotiented"] else float("nan")
        l2f = g["layer2"]["filtered_quotiented"]["mean"] if g["layer2"]["filtered_quotiented"] else float("nan")
        af = g["layer1"]["filtered_a"] if g["layer1"]["filtered_a"] else dict(mean=float("nan"), std=float("nan"))
        h_series.append(h_mean)
        print(f"  {grid:5d} {g['convergence']['n_well_converged']:5d} "
              f"{g['n_usable_gauge_pairs']:8d} {h_mean:12.4f} {c_mean:11.4f} "
              f"{l1f:10.4f} {l2f:10.4f} {af['mean']:9.3f} {af['std']:9.3f}")

    valid = [(g, h) for g, h in zip(GRIDS, h_series) if not np.isnan(h)]
    monotonic = all(valid[k][1] <= valid[k + 1][1] + 1e-12 for k in range(len(valid) - 1))
    print(f"\n  h-vs-affine residual by grid: "
          f"{ {g: (f'{h:.4f}' if not np.isnan(h) else 'n/a') for g, h in zip(GRIDS, h_series)} }")
    print(f"  monotonically increasing with grid: {monotonic}")

    if monotonic and len(valid) >= 3:
        verdict = "GRID_CONTROLS_GAUGE"
        verdict_text = ("h's distance from affine increases monotonically with spline grid size. "
                        "The near-affine h observed at grid=12 is (at least partly) an artifact of "
                        "spline discretization restricting which reparameterizations layer2 can "
                        "represent -- the d=1 gauge group is genuinely larger than affine, and a "
                        "finer grid lets training realize more of it.")
    else:
        verdict = "GRID_DOES_NOT_CONTROL_GAUGE"
        verdict_text = ("h's distance from affine does NOT increase monotonically with grid size. "
                        "The discretization explanation for why h came out near-affine at d=1 is "
                        "WRONG. Something else keeps trained reparameterizations close to affine -- "
                        "candidates: the SiLU base path (which is affine-in-shape and cannot be "
                        "reparameterized), optimization inductive bias, or the fact that both models "
                        "must fit the same smooth monotone target.")
    print(f"\nVERDICT: {verdict}\n  {verdict_text}")

    out = dict(grids=GRIDS, hidden=HIDDEN, seeds=SEEDS, r2_threshold=R2_THRESHOLD,
               per_grid=per_grid, h_series=h_series, monotonic=monotonic,
               verdict=verdict, verdict_text=verdict_text)
    with open(RESULTS_DIR / "kan_grid_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'kan_grid_sweep.json'}")


if __name__ == "__main__":
    main()
