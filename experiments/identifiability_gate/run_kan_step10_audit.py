#!/usr/bin/env python3
"""
Step 10 of experiments/kan-identifiability: BLOCKING audit of the d>=2 affine
result before it is written up. The Step 9 verdict flipped from "prediction
fails" to "confirmed" after a seed filter and a transform-type change; a result
that changes sign after mid-analysis revision has to be hardened.

No new experimental conditions are introduced. Models are deterministically
re-instantiated from the same seeds/hyperparameters already used (same code
path, forced CPU, identical seeds -> identical weights) purely so that layer
functions can be evaluated; the saved JSONs are reused for everything already
recorded.

10a DERIVED OR FITTED -- three layer2 tests, side by side:
    (i)   DERIVED-EXACT   phi2_i(u) vs phi2_j(h(u)), h = phi1_j.phi1_i^-1 built
          numerically from layer1. ZERO free parameters. (This is what Step 9
          reported.)
    (ii)  DERIVED-AFFINE  phi2_i(u) vs phi2_j(a*u+b), (a,b) TAKEN from layer1's
          own affine fit. ZERO free parameters on layer2. (The number 10a asks
          for: propagate layer1's scale/shift and PREDICT layer2.)
    (iii) FITTED-AFFINE   phi2_i(u) vs phi2_j(a*u+b), (a,b) fitted to minimize
          layer2's residual. TWO free parameters -- a generous upper bound on
          how well any input-domain affine map could do.

10b THE SLOW SEEDS -- per-seed final loss/R^2 at d>=2, every metric on all 10
    seeds alongside the 7-seed filtered numbers.
10c APPLES-TO-APPLES -- the same convergence filter AND the same transform-type
    correction applied to d=1, tabulated against d>=2.
10d SHARED-SCALE EFFECT SIZE -- within-pair vs across-pair spread of the fitted
    scale factor, and their ratio.

Usage:
    python experiments/identifiability_gate/run_kan_step10_audit.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from run_kan_seed_variance import (  # noqa: E402
    train_one_seed, raw_and_quotiented, summarize, SEEDS, N_EDGE_PTS, RESULTS_DIR,
)
from kan_composition_gauge_lib import (  # noqa: E402
    composition_gauge_test, make_marginal_layer_fn, make_layer_fn,
)
from run_kan_multivariate import train_one_seed_2d, X1_SCALE  # noqa: E402

R2_THRESHOLD = 0.999
D1_X_MIN, D1_X_MAX = 0.012, 1.0
D2_X1_MIN, D2_X1_MAX = 0.05 / X1_SCALE, 1.0


def derived_affine_residual(phi2_i_fn, phi2_j_fn, u_grid, a, b) -> float:
    """(ii) zero free parameters on layer2: (a,b) come from layer1's fit."""
    lhs = phi2_i_fn(u_grid)
    rhs = phi2_j_fn(a * u_grid + b)
    return float(np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + 1e-12))


def fitted_affine_residual(phi2_i_fn, phi2_j_fn, u_grid, a0, b0) -> tuple[float, float, float]:
    """(iii) two free parameters, fitted to minimize layer2's own residual."""
    lhs = phi2_i_fn(u_grid)
    denom = np.linalg.norm(lhs) + 1e-12

    def obj(p):
        return float(np.linalg.norm(lhs - phi2_j_fn(p[0] * u_grid + p[1])) / denom)

    best = minimize(obj, x0=np.array([a0, b0]), method="Nelder-Mead",
                    options=dict(maxiter=2000, xatol=1e-6, fatol=1e-10))
    return float(best.fun), float(best.x[0]), float(best.x[1])


def audit_regime(name: str, models: dict, well_converged: list[int],
                  phi1_factory, phi2_factory, x_min: float, x_max: float) -> list[dict]:
    """Run the three layer2 tests for every well-converged pair in one regime."""
    x_grid = np.linspace(x_min, x_max, N_EDGE_PTS)
    rows = []
    for i, j in itertools.combinations(well_converged, 2):
        phi1_i_fn, phi1_j_fn = phi1_factory(models[i]), phi1_factory(models[j])
        phi2_i_fn, phi2_j_fn = phi2_factory(models[i]), phi2_factory(models[j])

        phi1_i_vals = phi1_i_fn(x_grid)
        gauge = composition_gauge_test(x_grid, phi1_i_vals, phi1_j_fn, phi2_i_fn, phi2_j_fn)
        if not gauge["usable"]:
            rows.append(dict(regime=name, i=i, j=j, usable=False))
            continue

        v_grid = np.array(gauge["v_grid"])

        # layer1's OWN affine fit on the same raw curves, in the direction
        # phi1_j ~= a*phi1_i + b (so u -> a*u+b is how the hidden activation moves)
        phi1_j_vals = phi1_j_fn(x_grid)
        l1_fit = raw_and_quotiented(phi1_j_vals, phi1_i_vals)
        a_l1, b_l1 = l1_fit["a"], l1_fit["b"]

        derived_exact = gauge["composition_residual_raw"]
        derived_affine = derived_affine_residual(phi2_i_fn, phi2_j_fn, v_grid, a_l1, b_l1)
        fitted_aff, a_fit, b_fit = fitted_affine_residual(phi2_i_fn, phi2_j_fn, v_grid, a_l1, b_l1)

        rows.append(dict(
            regime=name, i=i, j=j, usable=True,
            layer1_affine_residual=l1_fit["quotiented"], layer1_a=a_l1, layer1_b=b_l1,
            h_affine_residual=gauge["h_affine_residual"],
            layer2_derived_exact=derived_exact,
            layer2_derived_affine=derived_affine,
            layer2_fitted_affine=fitted_aff, fitted_a=a_fit, fitted_b=b_fit,
        ))
        print(f"  {name} pair ({i},{j}): layer1_affine={l1_fit['quotiented']:.4f} (a={a_l1:.3f}) | "
              f"h-vs-affine={gauge['h_affine_residual']:.4f} | layer2: derived-exact="
              f"{derived_exact:.4f}  derived-affine={derived_affine:.4f}  fitted-affine={fitted_aff:.4f}")
    return rows


def mean_of(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get("usable")]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    out = {}

    # ── d=1 regime ──────────────────────────────────────────────────────────
    d1 = json.load(open(RESULTS_DIR / "kan_seed_variance_hidden1.json"))
    d1_r2 = {s["seed"]: s["test_r2"] for s in d1["per_seed"]}
    d1_wc = sorted(s for s in d1["seeds"] if d1_r2[s] > R2_THRESHOLD)
    print(f"d=1 (hidden=1): {len(d1_wc)}/{len(d1['seeds'])} well-converged: {d1_wc}")
    d1_models = {s: train_one_seed(s, hidden=1)["model"] for s in d1_wc}
    d1_rows = audit_regime(
        "d=1", d1_models, d1_wc,
        lambda m: make_marginal_layer_fn(m.layer1, in_features=1, input_idx=0),
        lambda m: make_layer_fn(m.layer2), D1_X_MIN, D1_X_MAX)

    # ── d>=2 regime ─────────────────────────────────────────────────────────
    d2 = json.load(open(RESULTS_DIR / "kan_multivariate.json"))
    d2_r2 = {s["seed"]: s["test_r2"] for s in d2["per_seed"]}
    d2_loss = {s["seed"]: s["final_train_loss"] for s in d2["per_seed"]}
    d2_wc = sorted(s for s in d2["seeds"] if d2_r2[s] > R2_THRESHOLD)
    d2_excluded = [s for s in d2["seeds"] if s not in d2_wc]
    print(f"\nd>=2: {len(d2_wc)}/{len(d2['seeds'])} well-converged: {d2_wc}  excluded: {d2_excluded}")
    print(f"Re-instantiating d>=2 models (deterministic, same seeds)...")
    d2_models = {s: train_one_seed_2d(s)["model"] for s in SEEDS}
    d2_rows = audit_regime(
        "d>=2", d2_models, d2_wc,
        lambda m: make_marginal_layer_fn(m.layer1, in_features=2, input_idx=0),
        lambda m: make_layer_fn(m.layer2), D2_X1_MIN, D2_X1_MAX)

    # ── 10b: all-10-seed d>=2 metrics alongside the 7-seed filtered ones ────
    print(f"\n=== 10b: THE SLOW SEEDS (d>=2, per-seed, 3000 epochs) ===")
    for s in d2["seeds"]:
        tag = "converged" if d2_r2[s] > R2_THRESHOLD else "NOT CONVERGED"
        print(f"  seed={s}: final_train_loss={d2_loss[s]:.3e}  test_R2={d2_r2[s]:.6f}   [{tag}]")
    all10_rows = audit_regime(
        "d>=2-all10", d2_models, sorted(d2["seeds"]),
        lambda m: make_marginal_layer_fn(m.layer1, in_features=2, input_idx=0),
        lambda m: make_layer_fn(m.layer2), D2_X1_MIN, D2_X1_MAX)

    d2_edges_all = {
        "layer1_x1": d2["layer1_x1_edge"]["quotiented_summary"]["mean"],
        "layer1_x2": d2["layer1_x2_edge"]["quotiented_summary"]["mean"],
        "layer2_output_affine": d2["layer2_edge"]["quotiented_summary"]["mean"],
    }
    d2_edges_filt = {k: d2["filtered_edge_comparison"][kk]["quotiented"]["mean"]
                      for k, kk in [("layer1_x1", "layer1_x1"), ("layer1_x2", "layer1_x2"),
                                     ("layer2_output_affine", "layer2")]}
    print(f"\n  all-10-seed (45 pairs) vs 7-seed filtered (21 pairs):")
    for k in d2_edges_all:
        print(f"    {k:22s}: all10={d2_edges_all[k]:.4f}   filtered={d2_edges_filt[k]:.4f}")
    print(f"    layer2 derived-exact  : all10={mean_of(all10_rows, 'layer2_derived_exact'):.4f}   "
          f"filtered={mean_of(d2_rows, 'layer2_derived_exact'):.4f}")
    print(f"    layer2 derived-affine : all10={mean_of(all10_rows, 'layer2_derived_affine'):.4f}   "
          f"filtered={mean_of(d2_rows, 'layer2_derived_affine'):.4f}")
    print(f"    h-vs-affine           : all10={mean_of(all10_rows, 'h_affine_residual'):.4f}   "
          f"filtered={mean_of(d2_rows, 'h_affine_residual'):.4f}")

    # ── 10c: apples-to-apples table ─────────────────────────────────────────
    d1_pair_idx = [(a, b) for a in range(len(d1["seeds"])) for b in range(len(d1["seeds"])) if a < b]
    d1_keep = [k for k, p in enumerate(d1_pair_idx) if p[0] in d1_wc and p[1] in d1_wc]
    d1_l1_filt = float(np.mean([d1["layer1_edges"]["0"]["pairs"][k]["quotiented"] for k in d1_keep]))
    d1_l2_filt_outaff = float(np.mean([d1["layer2_edges"]["0"]["pairs"][k]["quotiented"] for k in d1_keep]))

    print(f"\n=== 10c: APPLES-TO-APPLES (same filter, same transform-type correction) ===")
    hdr = (f"  {'regime':8s} {'n_wc':>5s} {'n_pairs':>8s} {'L1_QUOT':>9s} {'L2_outaff':>10s} "
           f"{'L2_derExact':>12s} {'L2_derAffine':>13s} {'L2_fitAffine':>13s} {'h_vs_affine':>12s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows_table = []
    for label, wc, npairs, l1q, l2out, rows in [
        ("d=1", d1_wc, len(d1_keep), d1_l1_filt, d1_l2_filt_outaff, d1_rows),
        ("d>=2", d2_wc, len(list(itertools.combinations(d2_wc, 2))),
         (d2_edges_filt["layer1_x1"] + d2_edges_filt["layer1_x2"]) / 2,
         d2_edges_filt["layer2_output_affine"], d2_rows),
    ]:
        r = dict(regime=label, n_well_converged=len(wc), n_pairs=npairs,
                 layer1_quotiented=l1q, layer2_output_affine=l2out,
                 layer2_derived_exact=mean_of(rows, "layer2_derived_exact"),
                 layer2_derived_affine=mean_of(rows, "layer2_derived_affine"),
                 layer2_fitted_affine=mean_of(rows, "layer2_fitted_affine"),
                 h_vs_affine=mean_of(rows, "h_affine_residual"))
        rows_table.append(r)
        print(f"  {label:8s} {len(wc):5d} {npairs:8d} {l1q:9.4f} {l2out:10.4f} "
              f"{r['layer2_derived_exact']:12.4f} {r['layer2_derived_affine']:13.4f} "
              f"{r['layer2_fitted_affine']:13.4f} {r['h_vs_affine']:12.4f}")

    contrast_h = rows_table[0]["h_vs_affine"] / (rows_table[1]["h_vs_affine"] + 1e-12)
    l2_der_d1, l2_der_d2 = rows_table[0]["layer2_derived_exact"], rows_table[1]["layer2_derived_exact"]
    print(f"\n  h-vs-affine contrast d=1/d>=2 = {contrast_h:.1f}x  (d=1 looser, as theory predicts)")
    print(f"  layer2 derived-exact: d=1={l2_der_d1:.4f}  d>=2={l2_der_d2:.4f}  -> "
          f"{'d=1 is BETTER explained (contrast REVERSES on this metric)' if l2_der_d1 < l2_der_d2 else 'd>=2 better'}")

    # ── 10d: shared-scale effect size ───────────────────────────────────────
    print(f"\n=== 10d: SHARED-SCALE EFFECT SIZE (d>=2) ===")
    pair_idx_d2 = [(a, b) for a in range(10) for b in range(10) if a < b]
    keep = [k for k, p in enumerate(pair_idx_d2) if p[0] in d2_wc and p[1] in d2_wc]
    a_x1 = np.array([d2["layer1_x1_edge"]["pairs"][k]["a"] for k in keep])
    a_x2 = np.array([d2["layer1_x2_edge"]["pairs"][k]["a"] for k in keep])
    within_abs = np.abs(a_x1 - a_x2)
    within_rel = within_abs / np.maximum(np.maximum(np.abs(a_x1), np.abs(a_x2)), 1e-9)
    a_pooled = np.concatenate([a_x1, a_x2])
    across_std = float(np.std(a_pooled))
    across_range = float(a_pooled.max() - a_pooled.min())
    ratio_std = float(np.mean(within_abs) / (across_std + 1e-12))
    ratio_range = float(np.mean(within_abs) / (across_range + 1e-12))
    print(f"  within-pair |a_x1 - a_x2|: mean_abs={np.mean(within_abs):.4f}  "
          f"mean_rel={np.mean(within_rel):.4f}  max_abs={within_abs.max():.4f}")
    print(f"  across-pair a: std={across_std:.4f}  range={across_range:.4f} "
          f"(min={a_pooled.min():.3f}, max={a_pooled.max():.3f})")
    print(f"  RATIO within/across: {ratio_std:.4f} (vs std)   {ratio_range:.4f} (vs range)")
    print(f"  -> within-pair agreement is {1/max(ratio_std,1e-12):.0f}x tighter than across-pair spread")

    out = dict(
        note="Step 10 audit. Models deterministically re-instantiated (same seeds); no new conditions.",
        q10a=dict(
            answer="DERIVED (zero free parameters). Step 9's reported layer2 number is "
                   "composition_residual_raw = ||phi2_i - phi2_j.h|| / ||phi2_i||, where h = "
                   "phi1_j.phi1_i^-1 is built numerically from layer1's own functions -- no "
                   "parameters are fitted on the layer2 side at all. The stricter 'propagate "
                   "layer1's fitted (a,b)' variant (derived-affine) and the generous 2-parameter "
                   "fitted-affine variant are computed alongside for bracketing.",
            d1_rows=d1_rows, d2_rows=d2_rows),
        q10b=dict(per_seed=[dict(seed=s, final_train_loss=d2_loss[s], test_r2=d2_r2[s],
                                  converged=bool(d2_r2[s] > R2_THRESHOLD)) for s in d2["seeds"]],
                   well_converged=d2_wc, excluded=d2_excluded,
                   all10_rows=all10_rows,
                   edges_all10=d2_edges_all, edges_filtered=d2_edges_filt),
        q10c=dict(table=rows_table, h_contrast_d1_over_d2=contrast_h,
                   layer2_derived_exact_d1=l2_der_d1, layer2_derived_exact_d2=l2_der_d2),
        q10d=dict(within_pair_mean_abs=float(np.mean(within_abs)),
                   within_pair_mean_rel=float(np.mean(within_rel)),
                   within_pair_max_abs=float(within_abs.max()),
                   across_pair_std=across_std, across_pair_range=across_range,
                   ratio_within_over_across_std=ratio_std,
                   ratio_within_over_across_range=ratio_range),
    )
    with open(RESULTS_DIR / "kan_step10_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'kan_step10_audit.json'}")


if __name__ == "__main__":
    main()
