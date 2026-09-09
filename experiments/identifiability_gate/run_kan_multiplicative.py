#!/usr/bin/env python3
"""
Step 12 of experiments/kan-identifiability: test the PRE-REGISTERED predictions
in PREREGISTERED_step12_multiplicative.md (committed 69edebb, before this file
existed and before any multiplicative data was generated).

Theory: for a multiplicative node T = prod_j psi_j(x_j), the surviving gauge is
the POWER maps h(u)=u^c (the only continuous solutions of h(uv)=h(u)h(v)), plus
the reciprocal torus psi_1 -> lambda*psi_1, psi_2 -> psi_2/lambda.

Target:  f(x1,x2) = exp(-x1) * sin(pi*x2),  x1~U(0.05,4.0), x2~U(0.05,0.95)
         strictly positive on that domain (so power fits are well-posed) and
         exactly multiplicatively separable. MRI magnitude-model shape: a decay
         factor times a modulation factor.
Arch:    MultKAN[2 -> 1 -> 1] -- T = psi_1(x1)*psi_2(x2) using the SAME
         B-spline + SiLU edge machinery as KANLinear (prod instead of sum),
         then KANLinear(1,1) as the outer g. hidden=1 mirrors Steps 8-9.

EDGE EXTRACTION differs from the additive steps in one way, stated for the
record: because the node multiplies, a marginal sweep with other inputs at 0
would zero the product, so psi_j is read out DIRECTLY from the layer (exact,
white-box) rather than inferred by sweep-minus-baseline. This is strictly more
accurate than the additive convention, not less.

Reuses raw_and_quotiented / summarize (no metric reimplementation).

Usage:
    python experiments/identifiability_gate/run_kan_multiplicative.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mhwf_pikan.core.fno_kan import KANLinear, b_spline_basis  # noqa: E402
from run_kan_seed_variance import (  # noqa: E402
    raw_and_quotiented, summarize, SEEDS, N_EDGE_PTS, KAN_GRID, KAN_K, KAN_LR, RESULTS_DIR,
)

N_TRAIN, N_TEST = 8_000, 2_000
TRAIN_EPOCHS_MULT = 3000     # d>=2 additive needed 3000; reported, not tuned per-outcome
X1_SCALE = 4.0
X2_MIN, X2_MAX = 0.05, 0.95
R2_THRESHOLD = 0.999          # same filter as every prior step

# thresholds fixed in the pre-registration
STABLE_THRESHOLD = 0.10
MIN_RETAINED_FRAC = 0.30
POS_EPS_FRAC = 1e-3


class MultKANNode(nn.Module):
    """T = prod_j psi_j(x_j) with psi_j = base_w_j*silu(x_j) + base_b_j + spline_j(tanh(x_j)).

    Same edge machinery as KANLinear (SiLU base path + B-spline on tanh-squashed
    input); the ONLY change is prod over inputs instead of sum. base_bias is
    initialized at 1.0 so the product is not degenerate at initialization.
    """

    def __init__(self, in_features: int, grid_size: int = KAN_GRID, spline_order: int = KAN_K):
        super().__init__()
        self.in_features = in_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        n_knots = grid_size + 2 * spline_order + 1
        grid = torch.linspace(-1, 1, n_knots).unsqueeze(0).repeat(in_features, 1)
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.randn(in_features) * 0.1)
        self.base_bias = nn.Parameter(torch.ones(in_features))
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.randn(in_features, n_basis) * 0.1)

    def edge_values(self, x: torch.Tensor) -> torch.Tensor:
        """psi_j(x_j) for every j -> [N, in_features] (the node's factors)."""
        base = self.base_weight * F.silu(x) + self.base_bias
        basis = b_spline_basis(torch.tanh(x), self.grid, self.spline_order)
        spline = torch.einsum("ib,nib->ni", self.spline_weight, basis)
        return base + spline

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge_values(x).prod(dim=1, keepdim=True)


class MultKAN(nn.Module):
    def __init__(self, in_features: int = 2, grid: int = KAN_GRID, k: int = KAN_K):
        super().__init__()
        self.node = MultKANNode(in_features, grid_size=grid, spline_order=k)
        self.layer2 = KANLinear(1, 1, grid_size=grid, spline_order=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.node(x))


def train_one_seed_mult(seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x1_tr = rng.uniform(0.05, X1_SCALE, N_TRAIN).astype(np.float32)
    x2_tr = rng.uniform(X2_MIN, X2_MAX, N_TRAIN).astype(np.float32)
    x1_te = rng.uniform(0.05, X1_SCALE, N_TEST).astype(np.float32)
    x2_te = rng.uniform(X2_MIN, X2_MAX, N_TEST).astype(np.float32)

    y_tr = np.exp(-x1_tr) * np.sin(np.pi * x2_tr)
    y_te = np.exp(-x1_te) * np.sin(np.pi * x2_te)

    X_tr = torch.from_numpy(np.stack([x1_tr / X1_SCALE, x2_tr], axis=1))
    X_te = torch.from_numpy(np.stack([x1_te / X1_SCALE, x2_te], axis=1))
    Y_tr, Y_te = torch.from_numpy(y_tr), torch.from_numpy(y_te)

    model = MultKAN()
    opt = torch.optim.Adam(model.parameters(), lr=KAN_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS_MULT,
                                                        eta_min=KAN_LR * 0.01)
    final_loss = None
    for _ in range(TRAIN_EPOCHS_MULT):
        model.train()
        opt.zero_grad()
        loss = F.mse_loss(model(X_tr).squeeze(-1), Y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        pred = model(X_te).squeeze(-1)
        test_mse = float(F.mse_loss(pred, Y_te).item())
        ss_res = float(((pred - Y_te) ** 2).sum().item())
        ss_tot = float(((Y_te - Y_te.mean()) ** 2).sum().item())
        test_r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return dict(seed=seed, model=model, final_train_loss=final_loss,
                test_mse=test_mse, test_r2=test_r2)


def edge_curve(model: MultKAN, input_idx: int, x_min: float, x_max: float,
                n_pts: int = N_EDGE_PTS) -> np.ndarray:
    """psi_{input_idx} read directly off the node (exact, white-box)."""
    x_vals = torch.linspace(x_min, x_max, n_pts)
    X = torch.zeros(n_pts, 2)
    X[:, input_idx] = x_vals
    with torch.no_grad():
        return model.node.edge_values(X)[:, input_idx].numpy()


def power_fit(psi_i: np.ndarray, psi_j: np.ndarray) -> dict:
    """Pre-registered rule 3: fit log|psi_j| = log|c| + p*log|psi_i| where BOTH
    are strictly positive and bounded away from zero; residual reported in the
    ORIGINAL space normalized by ||psi_j|| over the SAME sub-domain."""
    eps_i = POS_EPS_FRAC * np.max(np.abs(psi_i))
    eps_j = POS_EPS_FRAC * np.max(np.abs(psi_j))
    mask = (psi_i > eps_i) & (psi_j > eps_j)
    retained = float(mask.sum() / len(psi_i))
    if retained < MIN_RETAINED_FRAC:
        return dict(retained_frac=retained, inconclusive=True)

    li, lj = np.log(psi_i[mask]), np.log(psi_j[mask])
    A = np.stack([li, np.ones_like(li)], axis=1)
    coef, *_ = np.linalg.lstsq(A, lj, rcond=None)
    p, log_c = float(coef[0]), float(coef[1])
    c = float(np.exp(log_c))

    pred = c * np.power(psi_i[mask], p)
    resid = float(np.linalg.norm(psi_j[mask] - pred) / (np.linalg.norm(psi_j[mask]) + 1e-12))

    # matched-domain affine residual on the SAME sub-domain, for a fair head-to-head
    aff = raw_and_quotiented(psi_j[mask], psi_i[mask])
    return dict(retained_frac=retained, inconclusive=False, p=p, c=c,
                power_residual=resid, affine_residual_matched_domain=aff["quotiented"],
                affine_a_matched=aff["a"])


def main() -> None:
    print("Step 12 (PRE-REGISTERED: PREREGISTERED_step12_multiplicative.md @ 69edebb)")
    print(f"Target: f(x1,x2)=exp(-x1)*sin(pi*x2), x1~U(0.05,{X1_SCALE}), x2~U({X2_MIN},{X2_MAX})")
    print(f"Arch: MultKAN[2->1->1], grid={KAN_GRID}, k={KAN_K}, epochs={TRAIN_EPOCHS_MULT}\n")

    runs = []
    for s in SEEDS:
        r = train_one_seed_mult(s)
        runs.append(r)
        print(f"  seed={s:2d}  final_train_loss={r['final_train_loss']:.3e}  "
              f"test_mse={r['test_mse']:.3e}  test_r2={r['test_r2']:.6f}")

    r2_summary = summarize([r["test_r2"] for r in runs])
    wc = sorted(r["seed"] for r in runs if r["test_r2"] > R2_THRESHOLD)
    print(f"\nConvergence: R^2 mean={r2_summary['mean']:.6f} min={r2_summary['min']:.6f}  "
          f"well-converged {len(wc)}/{len(SEEDS)}: {wc}")

    models = {r["seed"]: r["model"] for r in runs}
    x1_min, x1_max = 0.05 / X1_SCALE, 1.0
    curves = {
        1: {s: edge_curve(models[s], 0, x1_min, x1_max) for s in SEEDS},
        2: {s: edge_curve(models[s], 1, X2_MIN, X2_MAX) for s in SEEDS},
    }

    pair_list = list(itertools.combinations(wc, 2)) if len(wc) >= 2 else []
    print(f"\n=== P1/P2: affine vs power quotient, {len(pair_list)} well-converged pairs ===")
    rows = []
    for i, j in pair_list:
        row = dict(i=i, j=j, edges={})
        for e in (1, 2):
            psi_i, psi_j = curves[e][i], curves[e][j]
            aff_full = raw_and_quotiented(psi_j, psi_i)   # P1: registered metric, full grid
            pw = power_fit(psi_i, psi_j)                   # P2
            row["edges"][e] = dict(affine_residual_full=aff_full["quotiented"],
                                    affine_a_full=aff_full["a"], **pw)
            tag = "INCONCLUSIVE" if pw.get("inconclusive") else (
                f"p={pw['p']:.3f} c={pw['c']:.3f} power={pw['power_residual']:.4f} "
                f"affine_matched={pw['affine_residual_matched_domain']:.4f}")
            print(f"  pair({i},{j}) edge{e}: affine_full={aff_full['quotiented']:.4f}  "
                  f"retained={pw['retained_frac']:.0%}  {tag}")
        rows.append(row)

    def collect(key: str, edge: int | None = None) -> list[float]:
        vals = []
        for r in rows:
            for e, d in r["edges"].items():
                if edge is not None and e != edge:
                    continue
                if d.get("inconclusive"):
                    continue
                if key in d:
                    vals.append(d[key])
        return vals

    aff_full_vals = collect("affine_residual_full")
    aff_matched_vals = collect("affine_residual_matched_domain")
    power_vals = collect("power_residual")
    n_inconclusive = sum(1 for r in rows for d in r["edges"].values() if d.get("inconclusive"))

    verdicts = {}
    if not power_vals:
        verdicts["P1"] = verdicts["P2"] = "INCONCLUSIVE"
        p1_txt = p2_txt = "no usable pairs/edges"
        aff_full_m = aff_matched_m = power_m = float("nan")
    else:
        aff_full_m = float(np.mean(aff_full_vals))
        aff_matched_m = float(np.mean(aff_matched_vals))
        power_m = float(np.mean(power_vals))
        verdicts["P1"] = "HELD" if aff_full_m >= STABLE_THRESHOLD else "FAILED"
        verdicts["P2"] = "HELD" if power_m < STABLE_THRESHOLD else "FAILED"
        p1_txt = (f"affine residual (registered, full grid) mean={aff_full_m:.4f} "
                  f"({'>=' if aff_full_m >= STABLE_THRESHOLD else '<'} {STABLE_THRESHOLD})")
        p2_txt = (f"power residual mean={power_m:.4f} "
                  f"({'<' if power_m < STABLE_THRESHOLD else '>='} {STABLE_THRESHOLD}); "
                  f"matched-domain affine for comparison={aff_matched_m:.4f}")

    # P3: is p shared across the node's two edges within a pair?
    p3_within, p_all = [], []
    for r in rows:
        d1, d2 = r["edges"][1], r["edges"][2]
        if d1.get("inconclusive") or d2.get("inconclusive"):
            continue
        p1v, p2v = d1["p"], d2["p"]
        p3_within.append(abs(p1v - p2v) / max(abs(p1v), abs(p2v), 1e-9))
        p_all.extend([p1v, p2v])
    if p3_within:
        p3_mean = float(np.mean(p3_within))
        p_across_std = float(np.std(p_all))
        p3_ratio = float(p3_mean * np.mean([abs(x) for x in p_all]) / (p_across_std + 1e-12))
        verdicts["P3"] = "HELD" if (p3_mean < 0.10 and p3_ratio < 0.5) else "FAILED"
        p3_txt = (f"within-pair relative |p1-p2| mean={p3_mean:.4f}; across-pair p std="
                  f"{p_across_std:.4f}; ratio={p3_ratio:.4f}")
    else:
        p3_mean = p_across_std = p3_ratio = float("nan")
        verdicts["P3"] = "INCONCLUSIVE"
        p3_txt = "no pair had both edges conclusive"

    # P4: reciprocal torus in the prefactors
    c1s = [r["edges"][1]["c"] for r in rows
           if not r["edges"][1].get("inconclusive") and not r["edges"][2].get("inconclusive")]
    c2s = [r["edges"][2]["c"] for r in rows
           if not r["edges"][1].get("inconclusive") and not r["edges"][2].get("inconclusive")]
    if len(c1s) >= 3:
        lc1, lc2 = np.log(np.abs(c1s)), np.log(np.abs(c2s))
        r_pearson = float(np.corrcoef(lc1, lc2)[0, 1])
        prod = np.array(c1s) * np.array(c2s)
        cv = lambda v: float(np.std(v) / (abs(np.mean(v)) + 1e-12))
        cv_prod, cv_c1, cv_c2 = cv(prod), cv(np.array(c1s)), cv(np.array(c2s))
        verdicts["P4"] = "HELD" if (r_pearson < -0.3 and cv_prod < cv_c1 and cv_prod < cv_c2) else "FAILED"
        p4_txt = (f"pearson r(log|c1|, log|c2|)={r_pearson:.4f} (predicted negative); "
                  f"CV(c1*c2)={cv_prod:.4f} vs CV(c1)={cv_c1:.4f}, CV(c2)={cv_c2:.4f}")
    else:
        r_pearson = cv_prod = cv_c1 = cv_c2 = float("nan")
        verdicts["P4"] = "INCONCLUSIVE"
        p4_txt = "fewer than 3 conclusive pairs"

    print(f"\n{'='*72}\nPRE-REGISTERED PREDICTION RESULTS\n{'='*72}")
    for k, txt in [("P1", p1_txt), ("P2", p2_txt), ("P3", p3_txt), ("P4", p4_txt)]:
        print(f"  {k}: {verdicts[k]:12s} {txt}")
    print(f"\n  inconclusive edge-fits (retained<{MIN_RETAINED_FRAC:.0%}): {n_inconclusive}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    gx1 = np.linspace(x1_min, x1_max, N_EDGE_PTS)
    gx2 = np.linspace(X2_MIN, X2_MAX, N_EDGE_PTS)
    for s in SEEDS:
        style = dict(alpha=0.8, lw=1.4) if s in wc else dict(alpha=0.25, lw=0.8, ls=":")
        axes[0].plot(gx1, curves[1][s], **style)
        axes[1].plot(gx2, curves[2][s], **style)
    if pair_list:
        i, j = pair_list[0]
        for e, ax_lbl in [(1, "edge1"), (2, "edge2")]:
            d = rows[0]["edges"][e]
            if not d.get("inconclusive"):
                psi_i, psi_j = curves[e][i], curves[e][j]
                m = (psi_i > POS_EPS_FRAC * psi_i.max()) & (psi_j > POS_EPS_FRAC * psi_j.max())
                axes[2].loglog(psi_i[m], psi_j[m], lw=2, label=f"{ax_lbl}: p={d['p']:.2f}")
        axes[2].set_xlabel("psi_i"); axes[2].set_ylabel("psi_j")
        axes[2].set_title(f"power-law check, pair({i},{j})"); axes[2].legend(fontsize=8)
    axes[0].set_title("node edge psi_1(x1), all seeds")
    axes[1].set_title("node edge psi_2(x2), all seeds")
    fig.suptitle("Step 12: multiplicative-node edges (dotted = not well-converged)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "kan_multiplicative_edges.png", dpi=130)
    print(f"\n  → {RESULTS_DIR / 'kan_multiplicative_edges.png'}")

    out = dict(
        preregistration="PREREGISTERED_step12_multiplicative.md (commit 69edebb)",
        target="exp(-x1)*sin(pi*x2)", architecture="MultKAN[2->1->1]",
        x1_domain=[0.05, X1_SCALE], x2_domain=[X2_MIN, X2_MAX],
        epochs=TRAIN_EPOCHS_MULT, grid=KAN_GRID, spline_order=KAN_K, seeds=SEEDS,
        per_seed=[dict(seed=r["seed"], final_train_loss=r["final_train_loss"],
                        test_mse=r["test_mse"], test_r2=r["test_r2"]) for r in runs],
        convergence=dict(test_r2=r2_summary, well_converged=wc, n_well_converged=len(wc)),
        pairs=rows, n_inconclusive_edge_fits=n_inconclusive,
        summary=dict(affine_full_mean=aff_full_m, affine_matched_domain_mean=aff_matched_m,
                      power_mean=power_m, p3_within_mean=p3_mean, p_across_std=p_across_std,
                      p3_ratio=p3_ratio, p4_pearson=r_pearson,
                      cv_c1c2=cv_prod, cv_c1=cv_c1, cv_c2=cv_c2),
        verdicts=verdicts,
        verdict_text=dict(P1=p1_txt, P2=p2_txt, P3=p3_txt, P4=p4_txt),
    )
    with open(RESULTS_DIR / "kan_multiplicative.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {RESULTS_DIR / 'kan_multiplicative.json'}")


if __name__ == "__main__":
    main()
