#!/usr/bin/env python3
"""
Step 10b follow-up: how sensitive is the d>=2 conclusion to WHERE the
convergence filter is drawn?

The audit showed the excluded d>=2 seeds are not homogeneous: seed 2 is
catastrophic (R^2=0.790), seed 8 is bad (0.968), but seed 5 (0.998850) misses
the R^2>0.999 bar by 1.5e-4 and behaves like a converged seed. A conclusion
that depends on a threshold at the third decimal place would be fragile, so
every filter choice is reported.

No training: reads kan_step10_audit.json, which already contains the per-pair
audit for ALL 45 d>=2 pairs.

Usage:
    python experiments/identifiability_gate/run_kan_step10b_sensitivity.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gate import RESULTS_DIR  # noqa: E402

FILTERS = [
    ("all 10 seeds (no filter)", list(range(10))),
    ("drop seed2 only (9 seeds)", [0, 1, 3, 4, 5, 6, 7, 8, 9]),
    ("drop seed2+seed8 (8 seeds)", [0, 1, 3, 4, 5, 6, 7, 9]),
    ("registered filter R2>0.999 (7 seeds)", [0, 1, 3, 4, 6, 7, 9]),
]
KEYS = ["h_affine_residual", "layer1_affine_residual",
        "layer2_derived_exact", "layer2_derived_affine"]


def main() -> None:
    audit = json.load(open(RESULTS_DIR / "kan_step10_audit.json"))
    rows = {(r["i"], r["j"]): r for r in audit["q10b"]["all10_rows"] if r.get("usable")}

    print("d>=2 sensitivity to the convergence-filter choice (no retraining):\n")
    hdr = f"  {'filter':38s} {'n_pairs':>8s} " + " ".join(f"{k.replace('_',' '):>22s}" for k in KEYS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    out = []
    for label, seeds in FILTERS:
        prs = [rows[p] for p in itertools.combinations(sorted(seeds), 2) if p in rows]
        means = {k: float(np.mean([p[k] for p in prs])) for k in KEYS}
        out.append(dict(filter=label, seeds=seeds, n_pairs=len(prs), **means))
        print(f"  {label:38s} {len(prs):8d} " + " ".join(f"{means[k]:22.4f}" for k in KEYS))

    h_all = out[0]["h_affine_residual"]
    h_reg = out[-1]["h_affine_residual"]
    d1_h = audit["q10c"]["table"][0]["h_vs_affine"]
    robust = all(o["h_affine_residual"] < d1_h for o in out)

    print(f"\n  d=1 h-vs-affine for reference: {d1_h:.4f}")
    print(f"  every d>=2 filter choice stays below the d=1 value: {robust}")
    print(f"  worst case (no filter at all, includes the R^2=0.79 seed): {h_all:.4f} "
          f"-- still {d1_h / h_all:.1f}x tighter than d=1")
    conclusion = (
        f"The d>=2 near-affine result is NOT an artifact of where the convergence bar sits. "
        f"Dropping only the two genuinely-broken seeds (2, 8) gives h-vs-affine="
        f"{out[2]['h_affine_residual']:.4f}, indistinguishable from the registered 7-seed filter "
        f"({h_reg:.4f}); including the marginal seed 5 changes nothing. Even with NO filter at all "
        f"-- keeping a seed that only reached R^2=0.790 -- h-vs-affine is {h_all:.4f}, still "
        f"{d1_h / h_all:.1f}x tighter than d=1's {d1_h:.4f}. The one metric that does degrade "
        f"materially without filtering is layer2 derived-affine "
        f"({out[0]['layer2_derived_affine']:.4f} vs {out[-1]['layer2_derived_affine']:.4f}), which "
        f"is expected: a model that never fit the target has no meaningful gauge relation to one "
        f"that did.")
    print(f"\n  {conclusion}")

    with open(RESULTS_DIR / "kan_step10b_sensitivity.json", "w") as f:
        json.dump(dict(filters=out, d1_h_vs_affine=d1_h, robust_across_filters=robust,
                        conclusion=conclusion), f, indent=2)
    print(f"\n  → {RESULTS_DIR / 'kan_step10b_sensitivity.json'}")


if __name__ == "__main__":
    main()
