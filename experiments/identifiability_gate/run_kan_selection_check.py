#!/usr/bin/env python3
"""
Step A: is the near-affine d=1 result conditioned on SELECTION?

Concern as posed: Step 8 measured the reparameterization h only on pairs where
phi1_i could be inverted (strictly monotone over a usable interval). If pairs
were rejected because phi1 was badly behaved, and badly-behaved phi1 is exactly
where a wild h would show up, then "near-affine at d=1" would be an artifact of
the inversion requirement rather than a property of the gauge.

WHAT THE FILES SAY ABOUT THE PREMISE (see DISCREPANCIES in the note):
  - d=1 (hidden=1, grid=12) had 3 of 10 seeds pass R^2>0.999, giving 3 possible
    pairs -- not 21. (7 converged seeds is the d>=2 run.)
  - The invertibility requirement rejected NOTHING: all 3 candidate pairs were
    usable (interval fractions 1.00, 1.00, 0.96), and when the gauge test was
    run over all 45 pairs with no convergence filter (kan_grid_confound.json,
    grid=12) it found 45/45 usable.
  So monotonicity-based selection did not operate. The selection that DID
  operate is the convergence filter: 42 of 45 pairs are excluded because at
  least one of their two seeds failed R^2>0.999.

This script tests that selection, which is the stronger one (93% of pairs
excluded), using an inversion-free divergence computable on every pair:

    D_ij = min_{a,b} || phi1_i - (a*phi1_j + b) || / || phi1_i ||

D_ij is exactly the QUOTIENTED metric already computed for all 45 layer1 pairs
in kan_seed_variance_hidden1.json, so no retraining or regeneration is needed.
The inversion-based h-vs-affine residual for all 45 pairs is likewise already in
kan_grid_confound.json (grid=12). Both are compared on retained vs excluded.

Usage:
    python experiments/identifiability_gate/run_kan_selection_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate import RESULTS_DIR  # noqa: E402

R2_THRESHOLD = 0.999
N_SEEDS = 10


def summarize(v: list[float]) -> dict:
    a = np.array(v, dtype=float)
    return dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), median=float(np.median(a)), max=float(a.max()))


def main() -> None:
    d1 = json.loads((RESULTS_DIR / "kan_seed_variance_hidden1.json").read_text())
    cg = json.loads((RESULTS_DIR / "kan_composition_gauge_d1.json").read_text())
    gc = json.loads((RESULTS_DIR / "kan_grid_confound.json").read_text())

    r2 = {s["seed"]: s["test_r2"] for s in d1["per_seed"]}
    converged = sorted(s for s in r2 if r2[s] > R2_THRESHOLD)
    pair_idx = [(i, j) for i in range(N_SEEDS) for j in range(N_SEEDS) if i < j]

    # D_ij for every pair == the committed QUOTIENTED metric on layer1
    D = {p: d1["layer1_edges"]["0"]["pairs"][k]["quotiented"] for k, p in enumerate(pair_idx)}
    # inversion-based h-vs-affine for every pair, no convergence filter, grid=12
    H = {(r["i"], r["j"]): r["h_affine_residual"] for r in gc["rows"] if r["grid"] == 12}
    usable = set(H)

    retained = [tuple(sorted((r["i"], r["j"]))) for r in cg["pair_results"]
                if r["gauge"].get("usable")]
    excluded = [p for p in pair_idx if p not in retained]

    # rejection-reason census
    reasons = {}
    for p in pair_idx:
        if p in retained:
            reasons[p] = "retained"
        elif p not in usable:
            reasons[p] = "phi1 not invertible (non-monotone / interval too short)"
        else:
            failed = [s for s in p if r2[s] <= R2_THRESHOLD]
            reasons[p] = f"convergence filter: seed(s) {failed} below R^2>{R2_THRESHOLD}"
    counts: dict[str, int] = {}
    for v in reasons.values():
        key = ("retained" if v == "retained"
               else "phi1 not invertible" if v.startswith("phi1")
               else "convergence filter")
        counts[key] = counts.get(key, 0) + 1

    print(f"d=1 (hidden=1, grid=12): converged seeds {converged} ({len(converged)}/10)")
    print(f"pairs: {len(pair_idx)} total, {len(retained)} retained, {len(excluded)} excluded")
    print(f"invertibility usable over ALL pairs: {len(usable)}/{len(pair_idx)}")
    print("\nrejection-reason census:")
    for k, v in sorted(counts.items()):
        print(f"  {k:24s} {v:3d}")

    D_ret = [D[p] for p in retained]
    D_exc = [D[p] for p in excluded]
    H_ret = [H[p] for p in retained if p in H]
    H_exc = [H[p] for p in excluded if p in H]
    sD_ret, sD_exc = summarize(D_ret), summarize(D_exc)
    sH_ret, sH_exc = summarize(H_ret), summarize(H_exc)

    print(f"\nD (inversion-free affine divergence of phi1), all {len(pair_idx)} pairs measurable:")
    print(f"  retained (n={sD_ret['n']:2d}): mean={sD_ret['mean']:.4f} median={sD_ret['median']:.4f} "
          f"sd={sD_ret['std']:.4f} range=[{sD_ret['min']:.4f}, {sD_ret['max']:.4f}]")
    print(f"  excluded (n={sD_exc['n']:2d}): mean={sD_exc['mean']:.4f} median={sD_exc['median']:.4f} "
          f"sd={sD_exc['std']:.4f} range=[{sD_exc['min']:.4f}, {sD_exc['max']:.4f}]")
    print(f"  ratio excluded/retained (mean) = {sD_exc['mean']/max(sD_ret['mean'],1e-12):.2f}x")

    print(f"\nh-vs-affine (inversion-based), grid=12, all pairs usable:")
    print(f"  retained (n={sH_ret['n']:2d}): mean={sH_ret['mean']:.4f} range=[{sH_ret['min']:.4f}, {sH_ret['max']:.4f}]")
    print(f"  excluded (n={sH_exc['n']:2d}): mean={sH_exc['mean']:.4f} range=[{sH_exc['min']:.4f}, {sH_exc['max']:.4f}]")
    print(f"  ratio excluded/retained (mean) = {sH_exc['mean']/max(sH_ret['mean'],1e-12):.2f}x")

    # Mann-Whitney U (exact-ish, no scipy dependency beyond ranking) on D
    allv = D_ret + D_exc
    order = np.argsort(allv)
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    n1 = len(D_ret)
    U = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2)
    U_max = float(n1 * len(D_exc))
    print(f"\nMann-Whitney U on D: U={U:.1f} of {U_max:.0f} "
          f"(U/U_max={U/U_max:.3f}; 0.5 = no separation, ->0 = retained systematically smaller)")

    inflation = sD_exc["mean"] / max(sD_ret["mean"], 1e-12)
    selection_matters = (inflation > 1.5) and (U / U_max < 0.35)

    if selection_matters:
        verdict = "SELECTION_CONDITIONS_THE_RESULT"
        text = (f"Excluded pairs are systematically more divergent than retained ones "
                f"(D mean {sD_exc['mean']:.4f} vs {sD_ret['mean']:.4f}, {inflation:.2f}x; "
                f"U/U_max={U/U_max:.3f}). The near-affine d=1 result is conditioned on selection "
                f"and every d=1 claim must be reported as holding only on the converged subset.")
    else:
        verdict = "SELECTION_DOES_NOT_EXPLAIN_THE_RESULT"
        text = (f"Retained and excluded pairs are not separated in the direction that would "
                f"explain away the finding (D mean {sD_exc['mean']:.4f} excluded vs "
                f"{sD_ret['mean']:.4f} retained, {inflation:.2f}x; U/U_max={U/U_max:.3f}). "
                f"Note separately that the invertibility requirement rejected 0 of 45 pairs at "
                f"this configuration, so inversion-based selection did not operate at all; the "
                f"only selection applied was the convergence filter.")

    print(f"\nVERDICT: {verdict}\n  {text}")

    out = dict(
        note="Step A selection check. Pure post-hoc analysis: D is the committed QUOTIENTED "
             "metric on layer1 (kan_seed_variance_hidden1.json), h-vs-affine over all pairs is "
             "from kan_grid_confound.json grid=12. No training or regeneration.",
        config="d=1, hidden=1, grid=12", r2_threshold=R2_THRESHOLD,
        converged_seeds=converged, n_pairs_total=len(pair_idx),
        retained_pairs=[list(p) for p in retained], n_excluded=len(excluded),
        n_usable_invertible=len(usable),
        rejection_reason_counts=counts,
        rejection_reason_per_pair={f"{i}-{j}": reasons[(i, j)] for i, j in pair_idx},
        D_all={f"{i}-{j}": D[(i, j)] for i, j in pair_idx},
        D_retained=sD_ret, D_excluded=sD_exc,
        D_inflation_excluded_over_retained=inflation,
        h_retained=sH_ret, h_excluded=sH_exc,
        h_inflation_excluded_over_retained=sH_exc["mean"] / max(sH_ret["mean"], 1e-12),
        mann_whitney_U=U, mann_whitney_U_max=U_max, mann_whitney_ratio=U / U_max,
        verdict=verdict, verdict_text=text,
    )
    (RESULTS_DIR / "kan_selection_check.json").write_text(json.dumps(out, indent=2))
    print(f"\n  → {RESULTS_DIR / 'kan_selection_check.json'}")


if __name__ == "__main__":
    main()
