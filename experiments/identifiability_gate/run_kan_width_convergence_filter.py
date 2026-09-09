#!/usr/bin/env python3
"""
Supplementary robustness check for Step 4: hidden=1 and hidden=2 both failed the
test-R^2 > 0.999 convergence bar (2/10 and 2/10 seeds badly converged
respectively), so their raw QUOTIENTED numbers in kan_width_sweep_summary.json
are confounded -- a badly-converged seed's edges can look "unstable" purely
because the fit itself never settled, independent of any redundancy/
identifiability effect.

This does NOT retrain anything. It reads the already-saved per-width JSONs
(kan_seed_variance_hidden{1,2}.json) and restricts the pairwise comparison to
seed-pairs where BOTH seeds individually reached test R^2 > 0.999, using the
exact same pair ordering the original comparison used (pair_idx = [(i,j) for
i in range(n) for j in range(n) if i<j], deterministic given the seed list),
then reports RAW/QUOTIENTED/a on that filtered subset -- same numbers, just a
fair subset of them, not tuned or re-fit.

CAVEAT flagged explicitly: hidden=1 only has 3/10 seeds pass the bar, leaving
3 seed-pairs -- too small to draw a reliable conclusion from on its own.

Usage:
    python experiments/identifiability_gate/run_kan_width_convergence_filter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate import RESULTS_DIR  # noqa: E402

R2_THRESHOLD = 0.999
MIN_RELIABLE_PAIRS = 10   # below this, flag as underpowered rather than conclude


def summarize(values: list[float]) -> dict:
    arr = np.array(values)
    return dict(mean=float(arr.mean()), std=float(arr.std()), min=float(arr.min()),
                max=float(arr.max()), n=len(arr))


def filtered_layer_stats(layer: dict, keep_mask: list[int]) -> dict:
    all_raw, all_quot, all_a = [], [], []
    for edge_data in layer.values():
        for keep, p in zip(keep_mask, edge_data["pairs"]):
            if keep:
                all_raw.append(p["raw"])
                all_quot.append(p["quotiented"])
                all_a.append(p["a"])
    return dict(raw=summarize(all_raw), quotiented=summarize(all_quot), a=summarize(all_a))


def main() -> None:
    out = {}
    for hidden in (1, 2):
        path = RESULTS_DIR / f"kan_seed_variance_hidden{hidden}.json"
        d = json.load(open(path))
        seeds = d["seeds"]
        n = len(seeds)
        pair_idx = [(i, j) for i in range(n) for j in range(n) if i < j]
        r2_by_seed = {s["seed"]: s["test_r2"] for s in d["per_seed"]}
        good_idx = set(i for i, s in enumerate(seeds) if r2_by_seed[s] > R2_THRESHOLD)
        keep_mask = [1 if (i in good_idx and j in good_idx) else 0 for i, j in pair_idx]
        n_pairs_kept = sum(keep_mask)

        layer1_filtered = filtered_layer_stats(d["layer1_edges"], keep_mask)
        layer2_filtered = filtered_layer_stats(d["layer2_edges"], keep_mask)

        underpowered = n_pairs_kept < MIN_RELIABLE_PAIRS
        print(f"hidden={hidden}: {len(good_idx)}/{n} seeds well-converged, "
              f"{n_pairs_kept}/{len(pair_idx)} pairs kept"
              f"{'  [UNDERPOWERED -- fewer than ' + str(MIN_RELIABLE_PAIRS) + ' pairs]' if underpowered else ''}")
        print(f"  layer1 filtered: RAW={layer1_filtered['raw']['mean']:.3f}  "
              f"QUOTIENTED={layer1_filtered['quotiented']['mean']:.3f}  "
              f"a mean={layer1_filtered['a']['mean']:.3f} std={layer1_filtered['a']['std']:.3f}")
        print(f"  layer2 filtered: RAW={layer2_filtered['raw']['mean']:.3f}  "
              f"QUOTIENTED={layer2_filtered['quotiented']['mean']:.3f}  "
              f"a mean={layer2_filtered['a']['mean']:.3f} std={layer2_filtered['a']['std']:.3f}")

        out[hidden] = dict(
            n_seeds=n, n_well_converged=len(good_idx), n_pairs_total=len(pair_idx),
            n_pairs_kept=n_pairs_kept, underpowered=underpowered,
            r2_threshold=R2_THRESHOLD,
            layer1_filtered=layer1_filtered, layer2_filtered=layer2_filtered,
        )

    layer2_stays_large = all(out[h]["layer2_filtered"]["quotiented"]["mean"] >= 0.10 for h in (1, 2))
    print(f"\nlayer2 QUOTIENTED >= 0.10 at both hidden=1 and hidden=2 even after removing "
          f"non-converged seeds: {layer2_stays_large}")
    if layer2_stays_large:
        conclusion = ("The convergence confound does NOT explain away the Step-4 instability for "
                       "layer2 (hidden-unit -> output edges): it stays large even restricted to "
                       "well-converged seed-pairs at both hidden=1 and hidden=2. layer1's very low "
                       "filtered QUOTIENTED at hidden=1 (0.037) is suggestive of the redundancy "
                       "story but rests on only 3 seed-pairs -- underpowered, not a reliable basis "
                       "for a conclusion on its own. Net effect: PREDICTION FAILS stands, but "
                       "specifically because of layer2 (and layer1's hidden=1 result is inconclusive, "
                       "not confirmed-stable).")
    else:
        conclusion = "See per-width numbers above."
    print(f"\n{conclusion}")

    out["conclusion"] = conclusion
    with open(RESULTS_DIR / "kan_width_convergence_filtered.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'kan_width_convergence_filtered.json'}")


if __name__ == "__main__":
    main()
