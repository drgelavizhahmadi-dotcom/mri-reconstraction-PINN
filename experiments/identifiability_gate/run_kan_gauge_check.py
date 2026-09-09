#!/usr/bin/env python3
"""
Step 1 (BLOCKING, no training) of experiments/kan-identifiability:
does scale-then-compensate gauge freedom actually exist in this repo's KANLinear,
and does it survive when only the SPLINE half of an edge (not its SiLU base path)
is scaled?

Loads the checkpoint produced by reproduce_kan_task_a.py (no training here).

MODEL: SmallKAN = layer2(layer1(x)), layer1: KANLinear(1,4), layer2: KANLinear(4,1).
Because layer1 has in_dim=1, each of its 4 hidden units has exactly ONE incoming
edge -- so "one KANLinear edge function" and "one whole hidden unit's pre-activation"
coincide for layer1, and layer2 plays the role of the theorem's downstream g exactly
(SmallKAN has no normalization/residual between the two layers to complicate this).

TEST A -- whole-edge scale (matches theorem: psi -> lambda*psi, g -> g(u/lambda)):
  For hidden unit o, scale ALL of layer1's parameters feeding it
  (base_weight[o,0], base_bias[o], spline_weight[o,0,:], spline_bias[o]) by lambda.
  Compose layer2 with 1/lambda by dividing layer1's output at index o by lambda
  before layer2 consumes it. Expected: exact restoration (<1e-5) -- this is the
  literal theorem transform, not a nontrivial architectural coincidence.

TEST B -- spline-only sub-scale (does the SiLU base path ride along?):
  Same edge o, but scale ONLY spline_weight[o,0,:] by lambda (base_weight/base_bias/
  spline_bias untouched). Apply the SAME downstream correction (divide index o by
  lambda). Expected: measurable deviation, because base(SiLU) and spline(B-spline)
  paths are summed within the same edge but only one was scaled -- exposes that
  edge-level gauge freedom in this hybrid architecture is NOT sub-divisible between
  its two additive paths.

Usage:
    python experiments/identifiability_gate/run_kan_gauge_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mhwf_pikan.core.fno_kan import KANLinear  # noqa: E402
from run_gate import RESULTS_DIR                # noqa: E402

CHECKPOINT_PATH = _REPO_ROOT / "checkpoints" / "kan_task_a_repro.pt"
LAMBDA = 3.0
TOL = 1e-5
N_EVAL = 500          # fixed batch for output-invariance check
N_EDGE_PTS = 300       # points for before/after edge plot


class SmallKAN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, grid: int, k: int) -> None:
        super().__init__()
        self.layer1 = KANLinear(in_dim, hidden, grid_size=grid, spline_order=k)
        self.layer2 = KANLinear(hidden, out_dim, grid_size=grid, spline_order=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


def load_model() -> SmallKAN:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found. Run reproduce_kan_task_a.py first "
            f"(Step 0/prerequisite, not part of Step 1 itself)."
        )
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = SmallKAN(cfg["in_dim"], cfg["hidden"], cfg["out_dim"], cfg["grid"], cfg["k"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def scaled_layer1_output(layer1: KANLinear, x_flat: torch.Tensor, edge_o: int,
                          lam: float, spline_only: bool) -> torch.Tensor:
    """Recompute layer1(x) with hidden unit `edge_o`'s parameters scaled by lam.

    spline_only=False -> TEST A (whole edge: base + spline both scaled)
    spline_only=True  -> TEST B (spline_weight only; base path untouched)
    """
    bw = layer1.base_weight.clone()
    bb = layer1.base_bias.clone()
    sw = layer1.spline_weight.clone()
    sb = layer1.spline_bias.clone()

    if not spline_only:
        bw[edge_o, :] *= lam
        bb[edge_o] *= lam
    sw[edge_o, :, :] *= lam
    sb[edge_o] *= lam

    base_output = torch.nn.functional.linear(torch.nn.functional.silu(x_flat), bw, bb)
    x_normalized = torch.tanh(x_flat)
    from mhwf_pikan.core.fno_kan import b_spline_basis
    basis = b_spline_basis(x_normalized, layer1.grid, layer1.spline_order)
    spline_output = torch.einsum('oib,nib->no', sw, basis) + sb
    return base_output + layer1.spline_scale * spline_output


def run_test(model: SmallKAN, x_batch: torch.Tensor, edge_o: int,
             spline_only: bool) -> dict:
    with torch.no_grad():
        baseline = model(x_batch)

        u_perturbed = scaled_layer1_output(model.layer1, x_batch, edge_o, LAMBDA, spline_only)
        u_corrected = u_perturbed.clone()
        u_corrected[:, edge_o] = u_corrected[:, edge_o] / LAMBDA

        out_corrected = model.layer2(u_corrected)

    max_abs_diff = (out_corrected - baseline).abs().max().item()
    return dict(
        edge_o=edge_o, spline_only=spline_only, lam=LAMBDA,
        max_abs_diff=max_abs_diff, passes=max_abs_diff < TOL,
        baseline_mean=baseline.mean().item(), baseline_std=baseline.std().item(),
    )


def extract_edge(layer1: KANLinear, edge_o: int, lam: float, spline_only: bool,
                  x_min: float, x_max: float, n_pts: int = N_EDGE_PTS):
    x_vals = torch.linspace(x_min, x_max, n_pts).unsqueeze(1)  # in_dim=1
    x0 = torch.zeros(1, 1)
    with torch.no_grad():
        y_before_full = layer1(x_vals)[:, edge_o].numpy()
        y_before_base = layer1(x0)[0, edge_o].item()
        y_after_full = scaled_layer1_output(layer1, x_vals, edge_o, lam, spline_only)[:, edge_o].numpy()
        y_after_base = scaled_layer1_output(layer1, x0, edge_o, lam, spline_only)[0, edge_o].item()
    return x_vals.squeeze(1).numpy(), y_before_full - y_before_base, y_after_full - y_after_base


def main() -> None:
    model, ckpt = load_model()
    print(f"Loaded {CHECKPOINT_PATH.name} (task={ckpt['task']}, test_mse={ckpt['task_mse']:.3e}, "
          f"no training performed in this script)")

    torch.manual_seed(123)
    x_batch = torch.rand(N_EVAL, 1) * (1.0 - 0.012) + 0.012  # matches training domain [0.012, 1.0]

    n_hidden = model.layer1.spline_weight.shape[0]
    results_A, results_B = [], []
    for o in range(n_hidden):
        results_A.append(run_test(model, x_batch, o, spline_only=False))
        results_B.append(run_test(model, x_batch, o, spline_only=True))

    print("\n=== TEST A: whole-edge scale (base + spline), downstream/lambda ===")
    for r in results_A:
        status = "PASS" if r["passes"] else "FAIL"
        print(f"  edge o={r['edge_o']}  max|Δoutput|={r['max_abs_diff']:.3e}  [{status}]")

    print("\n=== TEST B: spline-only scale, base path untouched, same downstream/lambda ===")
    for r in results_B:
        status = "PASS" if r["passes"] else "FAIL"
        print(f"  edge o={r['edge_o']}  max|Δoutput|={r['max_abs_diff']:.3e}  [{status}]")

    a_pass = all(r["passes"] for r in results_A)
    b_pass = all(r["passes"] for r in results_B)

    if a_pass:
        verdict_a = ("REAL, unconstrained gauge freedom at whole-edge granularity: "
                     "outputs unchanged to <1e-5 when an edge's full function "
                     "(base+spline) is scaled by λ and layer2 is composed with 1/λ. "
                     "This is the literal theorem transform for this architecture "
                     "(layer1 has in_dim=1, so edge ≡ hidden unit; layer2 is exactly "
                     "the downstream g, no normalization/residual in between).")
    else:
        verdict_a = ("UNEXPECTED: whole-edge scaling did NOT leave outputs invariant "
                     "-- architecture is not a clean 2-layer KAN composition even at "
                     "this granularity. STOP: see per-edge max|Δoutput| above.")

    if b_pass:
        verdict_b = ("UNEXPECTED: spline-only scaling also left outputs invariant -- "
                     "implies the SiLU base path contributes ~0 to these edges (check "
                     "base_weight magnitudes), so spline and base are not really both load-bearing.")
    else:
        verdict_b = ("Spline-only scaling breaks output invariance under the same "
                     "downstream correction: base(SiLU) and spline(B-spline) paths are "
                     "summed within one edge but only one was rescaled, so they are NOT "
                     "independently gauge-free. Edge-level gauge freedom in this hybrid "
                     "architecture applies only to the WHOLE edge (base+spline together), "
                     "never to the spline coefficients in isolation.")

    print(f"\nTEST A verdict: {verdict_a}")
    print(f"TEST B verdict: {verdict_b}")

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, n_hidden, figsize=(4 * n_hidden, 7), sharex=True)
    x_min, x_max = 0.012, 1.0
    for o in range(n_hidden):
        x, y_before, y_after_A = extract_edge(model.layer1, o, LAMBDA, spline_only=False,
                                               x_min=x_min, x_max=x_max)
        _, _, y_after_B = extract_edge(model.layer1, o, LAMBDA, spline_only=True,
                                        x_min=x_min, x_max=x_max)
        axes[0, o].plot(x, y_before, label="before", lw=2)
        axes[0, o].plot(x, y_after_A, "--", label=f"after (whole-edge ×{LAMBDA:g})", lw=2)
        axes[0, o].set_title(f"edge o={o}: TEST A")
        axes[0, o].legend(fontsize=7)

        axes[1, o].plot(x, y_before, label="before", lw=2)
        axes[1, o].plot(x, y_after_B, "--", label=f"after (spline-only ×{LAMBDA:g})", lw=2)
        axes[1, o].set_title(f"edge o={o}: TEST B")
        axes[1, o].legend(fontsize=7)
    fig.suptitle("Layer-1 edge functions before/after gauge scaling (Step 1, kan-identifiability)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "kan_gauge_check_edges.png", dpi=130)
    print(f"  → {RESULTS_DIR / 'kan_gauge_check_edges.png'}")

    out = dict(
        lambda_=LAMBDA, tolerance=TOL, n_eval_batch=N_EVAL,
        checkpoint=str(CHECKPOINT_PATH.relative_to(_REPO_ROOT)),
        test_A_whole_edge=results_A, test_B_spline_only=results_B,
        test_A_all_pass=a_pass, test_B_all_pass=b_pass,
        verdict_A=verdict_a, verdict_B=verdict_b,
    )
    with open(RESULTS_DIR / "kan_gauge_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {RESULTS_DIR / 'kan_gauge_check.json'}")


if __name__ == "__main__":
    main()
