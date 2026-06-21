#!/usr/bin/env python3
"""
Stress-test: is MC-dropout uncertainty trustworthy beyond marginal coverage?
=============================================================================
The previous calibration run (run_calib_uncertainty.py) showed MC-dropout is
OVERCONFIDENT (cov50≈7.5%, should be 50%) but spatially ranking-informative
(Spearman ρ≈0.32, boundary/interior ratio≈1.6×).

Three targeted checks:

CHECK 1 — Conditional coverage
    Find global temperature T on held-out set such that marginal 50% / 90%
    coverage is restored.  Then report coverage SEPARATELY for:
      (a) boundary pixels (top-20% Sobel gradient on GT T2*)
      (b) interior pixels (bottom-50% gradient)
      (c) high-error pixels (top-50% |T2_rec − T2_gt| on held-out set)
      (d) low-error pixels (bottom-50%)
    Question: does the global T that fixes marginal coverage also fix
    conditional coverage, or do boundaries / high-error pixels stay
    under-covered?

CHECK 2 — Temperature transfer
    T is fitted at R=4 for 50% marginal coverage.
    Apply it UNCHANGED at R=2 and R=8.
    Report marginal + conditional coverage at each R.
    Question: is T regime-specific, or does one scalar transfer?

CHECK 3 — Ranking invariance
    Report Spearman ρ before and after temperature scaling.
    Expected: ρ is invariant (monotone scale preserves rank-order).
    This establishes ranking as the knob-invariant signal.

VERDICT:
    CALIBRATED_INTERVALS  — if conditional coverage ≥ nominal at all subsets
                            AND T transfers across R.
    RANKING-CALIBRATED_ABSTENTION — if conditional coverage fails
                            (boundaries under-covered) OR T does not transfer.

Usage:
    python experiments/identifiability_gate/run_stress_calibration.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import sobel
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_gate import (                                          # noqa: E402
    synthesise, H, W, TEs_MS, N_ECHOES, CF, HIDDEN,
    T2_MIN, T2_MAX, RESULTS_DIR,
)
from run_offres_fix import (                                    # noqa: E402
    DF_BOUND, _make_complex_zf_input,
)
from run_calib_uncertainty import (                             # noqa: E402
    BottleneckNet3MC, train_mc_model, mc_predict,
    load_test_maps, spearman_corr, coverage,
    ACCEL_SWEEP, Z_50, Z_90, N_MC, TRAIN_EPOCHS, DROPOUT_P,
    MASK_SEED, NOISE_SEED, SNR_DB,
)

warnings.filterwarnings("ignore")

# ─────────────────────────── constants ───────────────────────────────────────

R_CALIB   = 4                 # temperature fitted at this R
SEED      = 42                # reproducibility
N_BINS_COND = 5               # bins for conditional coverage reliability diagram


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — BUILD DATA FOR ONE MAP AT ONE R
# ═══════════════════════════════════════════════════════════════════════════════

def _make_zf_input(S0: np.ndarray, T2: np.ndarray,
                   R: int, device: torch.device,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicates the exact data-prep from run_calib_uncertainty.py run_one_map_one_R:
    synthesise → mask dict → ZF input tensor.
    """
    data     = synthesise(S0, T2, TEs_MS, accel=R, cf=CF,
                          snr_db=SNR_DB, mask_seed=MASK_SEED, noise_seed=NOISE_SEED)
    mask_tc  = torch.from_numpy(data["mask_2d"])
    k_under  = torch.from_numpy(data["kspace_under"]).to(torch.complex64)
    zf_arr   = _make_complex_zf_input(data["kspace_under"])
    zf_input = torch.from_numpy(zf_arr).unsqueeze(0)     # [1, 2*N_E, H, W]
    return zf_input, k_under, mask_tc


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — TEMPERATURE SCALING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def fit_temperature(
    std_raw: np.ndarray, abs_err: np.ndarray, fg: np.ndarray,
    target_coverage: float = 0.50,
) -> float:
    """
    Find scalar T so that coverage(T*std_raw, z_α) = target_coverage.

    Derivation:
        P(|err| ≤ z·T·std) = target
        ⟺ P(|err|/std ≤ z·T) = target
        ⟺ z·T = quantile(|err|/std, target)
        ⟺ T = quantile(|err|/std, target) / z
    """
    z = Z_50 if target_coverage == 0.50 else Z_90
    r = abs_err[fg] / (std_raw[fg] + 1e-8)          # calibration ratio
    T = float(np.quantile(r, target_coverage) / z)
    return T


def conditional_coverage(
    std_scaled: np.ndarray, abs_err: np.ndarray,
    fg_mask: np.ndarray, z: float,
) -> float:
    """Coverage restricted to fg_mask pixels only."""
    if fg_mask.sum() == 0:
        return float("nan")
    covered = abs_err[fg_mask] <= z * std_scaled[fg_mask]
    return float(covered.mean())


def _gradient_masks(T2_gt: np.ndarray, fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (interior_mask, boundary_mask) as 1D fg-pixel boolean arrays."""
    gx   = sobel(T2_gt.astype(np.float64), axis=0)
    gy   = sobel(T2_gt.astype(np.float64), axis=1)
    grad = np.sqrt(gx**2 + gy**2).astype(np.float32)

    fg_grad = grad[fg]
    p50     = np.percentile(fg_grad, 50)
    p80     = np.percentile(fg_grad, 80)

    interior = np.zeros_like(fg)
    boundary = np.zeros_like(fg)
    fg_idx   = np.where(fg.ravel())[0]

    int_px = fg_idx[fg_grad <= p50]
    bnd_px = fg_idx[fg_grad >= p80]
    interior.ravel()[int_px] = True
    boundary.ravel()[bnd_px] = True
    return interior, boundary


def _error_split_masks(abs_err: np.ndarray, fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (low_error_mask, high_error_mask) split at median fg error."""
    fg_err    = abs_err[fg]
    med       = np.median(fg_err)
    fg_idx    = np.where(fg.ravel())[0]
    low_mask  = np.zeros_like(fg); high_mask = np.zeros_like(fg)
    low_mask.ravel()[fg_idx[fg_err <= med]]  = True
    high_mask.ravel()[fg_idx[fg_err  > med]] = True
    return low_mask, high_mask


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — RUN ONE MAP AT ONE R (train + MC inference)
# ═══════════════════════════════════════════════════════════════════════════════

def run_one(tmap: dict, R: int, device: torch.device) -> dict:
    """Train MC-dropout model for one map at one R, return pixel-level arrays."""
    torch.manual_seed(SEED)
    S0, T2_gt, fg = tmap["S0"], tmap["T2"], tmap["fg"]
    zf_input, k_under_tc, mask_tc = _make_zf_input(S0, T2_gt, R, device)

    model = BottleneckNet3MC(
        n_echoes=N_ECHOES, hidden=HIDDEN,
        T2_min=T2_MIN, T2_max=T2_MAX,
        df_bound=DF_BOUND, p=DROPOUT_P,
    )
    label = f"R={R} {tmap['file'][-12:]} sl{tmap['slice']} s{tmap['seed']}"
    model, _ = train_mc_model(model, zf_input, k_under_tc, mask_tc,
                              TRAIN_EPOCHS, device, label)

    T2_mean, T2_std = mc_predict(model, zf_input, device, n_samples=N_MC)
    abs_err = np.abs(T2_mean - T2_gt)

    return dict(
        T2_gt   = T2_gt,
        T2_mean = T2_mean,
        T2_std  = T2_std,       # raw (un-scaled) std
        abs_err = abs_err,
        fg      = fg,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check1_conditional(
    per_map: list[dict], T_50: float, T_90: float
) -> dict:
    """
    CHECK 1: conditional coverage after global temperature scaling.

    T_50 chosen to restore 50% marginal coverage on the aggregated held-out set.
    """
    print(f"\n  [CHECK 1] T_50={T_50:.4f}  T_90={T_90:.4f}")

    rows = {}
    for z_label, z, T in [("50%", Z_50, T_50), ("90%", Z_90, T_50)]:
        # Aggregate pixel arrays across all maps
        all_std    = np.concatenate([m["T2_std"][m["fg"]]  for m in per_map])
        all_err    = np.concatenate([m["abs_err"][m["fg"]] for m in per_map])
        all_std_sc = all_std * T

        # Overall marginal
        n_fg  = len(all_err)
        marg  = float((all_err <= z * all_std_sc).mean())

        # Boundary/interior (need per-map then pool)
        cov_bnd  = []; cov_int = []; cov_hi = []; cov_lo = []
        for m in per_map:
            fg       = m["fg"]
            std_sc   = m["T2_std"] * T
            err      = m["abs_err"]

            int_m, bnd_m = _gradient_masks(m["T2_gt"], fg)
            lo_m,  hi_m  = _error_split_masks(err, fg)

            if int_m.any(): cov_int.append(conditional_coverage(std_sc, err, int_m, z))
            if bnd_m.any(): cov_bnd.append(conditional_coverage(std_sc, err, bnd_m, z))
            if lo_m.any():  cov_lo.append(conditional_coverage(std_sc, err, lo_m, z))
            if hi_m.any():  cov_hi.append(conditional_coverage(std_sc, err, hi_m, z))

        rows[z_label] = dict(
            marginal  = marg,
            interior  = float(np.mean(cov_int)) if cov_int else float("nan"),
            boundary  = float(np.mean(cov_bnd)) if cov_bnd else float("nan"),
            low_err   = float(np.mean(cov_lo))  if cov_lo  else float("nan"),
            high_err  = float(np.mean(cov_hi))  if cov_hi  else float("nan"),
        )

        print(f"    [{z_label}] marginal={marg:.3f}  "
              f"interior={rows[z_label]['interior']:.3f}  "
              f"boundary={rows[z_label]['boundary']:.3f}  "
              f"low_err={rows[z_label]['low_err']:.3f}  "
              f"high_err={rows[z_label]['high_err']:.3f}")

        deficit_bnd = rows[z_label]["boundary"] - float(z_label[:-1]) / 100.0
        deficit_hi  = rows[z_label]["high_err"] - float(z_label[:-1]) / 100.0
        print(f"           Δ(boundary-nominal)={deficit_bnd:+.3f}  "
              f"Δ(high_err-nominal)={deficit_hi:+.3f}")

    return rows


def check2_transfer(
    per_map_by_R: dict[int, list[dict]], T_calib: float
) -> dict:
    """
    CHECK 2: apply T fitted at R_CALIB unchanged to other Rs.
    Returns coverage table for marginal + conditional at each R.
    """
    print(f"\n  [CHECK 2] Temperature from R={R_CALIB}: T={T_calib:.4f}")
    result = {}
    for R, maps in sorted(per_map_by_R.items()):
        row = {}
        for z_label, z in [("50%", Z_50), ("90%", Z_90)]:
            cov_marg = []; cov_bnd = []; cov_int = []
            for m in maps:
                fg      = m["fg"]
                std_sc  = m["T2_std"] * T_calib
                err     = m["abs_err"]
                cov_marg.append(float((err[fg] <= z * std_sc[fg]).mean()))
                int_m, bnd_m = _gradient_masks(m["T2_gt"], fg)
                if int_m.any(): cov_int.append(conditional_coverage(std_sc, err, int_m, z))
                if bnd_m.any(): cov_bnd.append(conditional_coverage(std_sc, err, bnd_m, z))

            row[z_label] = dict(
                marginal = float(np.mean(cov_marg)),
                interior = float(np.mean(cov_int)) if cov_int else float("nan"),
                boundary = float(np.mean(cov_bnd)) if cov_bnd else float("nan"),
            )
        result[R] = row
        print(f"    R={R}: "
              f"50%→marg={result[R]['50%']['marginal']:.3f} "
              f"bnd={result[R]['50%']['boundary']:.3f}  "
              f"90%→marg={result[R]['90%']['marginal']:.3f} "
              f"bnd={result[R]['90%']['boundary']:.3f}")
    return result


def check3_ranking(per_map_by_R: dict[int, list[dict]], T_calib: float) -> dict:
    """
    CHECK 3: Spearman ρ before vs after temperature scaling (must be identical).
    """
    print(f"\n  [CHECK 3] Ranking invariance under temperature scaling T={T_calib:.4f}")
    result = {}
    for R, maps in sorted(per_map_by_R.items()):
        stds_raw  = np.concatenate([m["T2_std"][m["fg"]]  for m in maps])
        stds_scl  = stds_raw * T_calib
        errs      = np.concatenate([m["abs_err"][m["fg"]] for m in maps])

        rho_raw, _ = spearmanr(stds_raw, errs)
        rho_scl, _ = spearmanr(stds_scl, errs)
        delta_rho  = abs(rho_scl - rho_raw)

        result[R] = dict(rho_raw=float(rho_raw), rho_scaled=float(rho_scl),
                         delta=float(delta_rho))
        print(f"    R={R}: ρ_raw={rho_raw:.6f}  ρ_scaled={rho_scl:.6f}  "
              f"Δ={delta_rho:.2e}  ({'INVARIANT' if delta_rho < 1e-4 else 'CHANGED'})")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def save_figures(
    check1: dict,
    check2: dict,
    check3: dict,
    T_50: float, T_90: float,
) -> None:
    """
    Figure A: conditional coverage bar chart (CHECK 1 + 2 combined).
    Figure B: ranking invariance scatter (CHECK 3).
    """
    # ── Figure A: conditional coverage ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"MC-dropout calibration stress-test\n"
        f"Temperature T={T_50:.3f} (fitted at R={R_CALIB} for 50% marginal coverage)",
        fontsize=10,
    )

    # Left: CHECK 1 — conditional coverage at R=4 after temperature scaling
    ax = axes[0]
    groups   = ["marginal", "interior", "boundary", "low_err", "high_err"]
    labels   = ["Marginal", "Interior\n(bot-50%\ngrad)", "Boundary\n(top-20%\ngrad)",
                "Low-error\n(bot-50%)", "High-error\n(top-50%)"]
    nominal  = {"50%": 0.50, "90%": 0.90}
    x        = np.arange(len(groups))
    width    = 0.35
    c1_50    = check1.get("50%", {})
    c1_90    = check1.get("90%", {})
    vals_50  = [c1_50.get(g, float("nan")) for g in groups]
    vals_90  = [c1_90.get(g, float("nan")) for g in groups]
    bars1 = ax.bar(x - width / 2, vals_50, width, label="50% nominal", color="steelblue")
    bars2 = ax.bar(x + width / 2, vals_90, width, label="90% nominal", color="tomato", alpha=0.8)
    ax.axhline(0.50, ls="--", lw=1.2, color="steelblue", alpha=0.7)
    ax.axhline(0.90, ls="--", lw=1.2, color="tomato",    alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Coverage fraction")
    ax.set_title(f"CHECK 1: conditional coverage (R={R_CALIB}, T={T_50:.3f})\n"
                 f"Dashed = nominal target")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7)

    # Right: CHECK 2 — temperature transfer across Rs
    ax2  = axes[1]
    Rs   = sorted(check2.keys())
    marg_50 = [check2[R]["50%"]["marginal"] for R in Rs]
    bnd_50  = [check2[R]["50%"]["boundary"] for R in Rs]
    marg_90 = [check2[R]["90%"]["marginal"] for R in Rs]
    bnd_90  = [check2[R]["90%"]["boundary"] for R in Rs]

    x2 = np.arange(len(Rs)); w2 = 0.20
    ax2.bar(x2 - 1.5*w2, marg_50, w2, label="50% marg", color="steelblue")
    ax2.bar(x2 - 0.5*w2, bnd_50,  w2, label="50% bnd",  color="steelblue", alpha=0.5, hatch="//")
    ax2.bar(x2 + 0.5*w2, marg_90, w2, label="90% marg", color="tomato")
    ax2.bar(x2 + 1.5*w2, bnd_90,  w2, label="90% bnd",  color="tomato",    alpha=0.5, hatch="//")
    ax2.axhline(0.50, ls="--", lw=1.2, color="steelblue", alpha=0.7)
    ax2.axhline(0.90, ls="--", lw=1.2, color="tomato",    alpha=0.7)
    ax2.set_xticks(x2); ax2.set_xticklabels([f"R={R}" for R in Rs])
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Coverage fraction")
    ax2.set_title(f"CHECK 2: temperature transfer (T={T_50:.3f} from R={R_CALIB})\n"
                  f"Solid=marginal, hatched=boundary, dashed=nominal")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = RESULTS_DIR / "stress_calibration.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\n  → {out.name}")

    # ── Figure B: ranking invariance (CHECK 3) ─────────────────────────────────
    fig2, ax3 = plt.subplots(1, 1, figsize=(6, 4))
    Rs3 = sorted(check3.keys())
    rho_raw = [check3[R]["rho_raw"]    for R in Rs3]
    rho_scl = [check3[R]["rho_scaled"] for R in Rs3]
    w_b = 0.35
    x3  = np.arange(len(Rs3))
    ax3.bar(x3 - w_b/2, rho_raw, w_b, label="ρ raw (no T)",  color="steelblue")
    ax3.bar(x3 + w_b/2, rho_scl, w_b, label=f"ρ scaled (×T={T_50:.3f})", color="orange")
    ax3.set_xticks(x3); ax3.set_xticklabels([f"R={R}" for R in Rs3])
    ax3.set_ylabel("Spearman ρ (std vs |error|)")
    ax3.set_title("CHECK 3: Ranking invariance under temperature scaling\n"
                  "Bars should be identical (monotone T preserves rank-order)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(0, max(rho_raw + rho_scl) * 1.2 + 0.05)

    fig2.tight_layout()
    out2 = RESULTS_DIR / "stress_ranking.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight"); plt.close(fig2)
    print(f"  → {out2.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def print_verdict(
    check1: dict, check2: dict, check3: dict,
    T_50: float, T_90: float,
) -> str:
    import textwrap

    c1_50 = check1.get("50%", {})
    c1_90 = check1.get("90%", {})

    # Conditional coverage gaps
    bnd_deficit_50  = c1_50.get("boundary",  0.0) - 0.50
    hi_deficit_50   = c1_50.get("high_err",  0.0) - 0.50
    bnd_deficit_90  = c1_90.get("boundary",  0.0) - 0.90
    hi_deficit_90   = c1_90.get("high_err",  0.0) - 0.90
    conditional_ok  = (bnd_deficit_50 >= -0.05 and hi_deficit_50 >= -0.05 and
                       bnd_deficit_90 >= -0.05 and hi_deficit_90 >= -0.05)

    # Transfer: does T from R=4 give acceptable 50% coverage at other Rs?
    Rs_other  = [R for R in sorted(check2.keys()) if R != R_CALIB]
    xfer_gaps = {R: check2[R]["50%"]["marginal"] - 0.50 for R in Rs_other}
    transfer_ok = all(abs(g) <= 0.10 for g in xfer_gaps.values())

    # Ranking invariance
    max_rho_delta = max(check3[R]["delta"] for R in check3)
    ranking_invariant = max_rho_delta < 1e-4

    # Consistency of T across coverage levels
    T_consistency_ok = abs(T_50 - T_90) / (T_50 + 1e-8) < 0.30

    print("\n" + "═" * 110)
    print("STRESS-CALIBRATION VERDICT")
    print("═" * 110)

    # ── Check 1 table ──
    print(f"\n  CHECK 1 — CONDITIONAL COVERAGE (T={T_50:.4f} fitted at R={R_CALIB})")
    print(f"  {'Subset':<18} {'50% cov':>8}  {'Δ(50%)':>8}  {'90% cov':>8}  {'Δ(90%)':>8}")
    print(f"  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for key, label in [("marginal","Marginal"), ("interior","Interior"),
                       ("boundary","Boundary"), ("low_err","Low-error"),
                       ("high_err","High-error")]:
        v50 = c1_50.get(key, float("nan"))
        v90 = c1_90.get(key, float("nan"))
        d50 = v50 - 0.50; d90 = v90 - 0.90
        flag = " ←UNDER" if (d50 < -0.05 or d90 < -0.05) else ""
        print(f"  {label:<18}  {v50:>8.3f}  {d50:>+8.3f}  {v90:>8.3f}  {d90:>+8.3f}{flag}")

    # ── Check 2 table ──
    print(f"\n  CHECK 2 — TEMPERATURE TRANSFER (T={T_50:.4f} from R={R_CALIB})")
    print(f"  {'R':>4}  {'50% marg':>10}  {'50% bnd':>10}  {'90% marg':>10}  {'90% bnd':>10}  {'Δ_marg_50%':>12}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")
    for R in sorted(check2.keys()):
        m50 = check2[R]["50%"]["marginal"]; b50 = check2[R]["50%"]["boundary"]
        m90 = check2[R]["90%"]["marginal"]; b90 = check2[R]["90%"]["boundary"]
        d   = m50 - 0.50
        tag = " *CALIB*" if R == R_CALIB else ("  ←DRIFT" if abs(d) > 0.10 else "")
        print(f"  {R:>4}  {m50:>10.3f}  {b50:>10.3f}  {m90:>10.3f}  {b90:>10.3f}  {d:>+12.3f}{tag}")

    # ── Check 3 table ──
    print(f"\n  CHECK 3 — RANKING INVARIANCE (max |Δρ| should be ≈0)")
    print(f"  {'R':>4}  {'ρ raw':>8}  {'ρ scaled':>10}  {'|Δρ|':>8}  {'Invariant?':>12}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*12}")
    for R in sorted(check3.keys()):
        r = check3[R]
        inv = "✓" if r["delta"] < 1e-4 else "✗"
        print(f"  {R:>4}  {r['rho_raw']:>8.6f}  {r['rho_scaled']:>10.6f}  "
              f"{r['delta']:>8.2e}  {inv:>12}")

    # ── Criteria ──
    print(f"\n  CRITERIA:")
    print(f"    C1a: boundary coverage ≥ nominal−5pp  (50%: Δ={bnd_deficit_50:+.3f}):  "
          f"{'✓' if bnd_deficit_50 >= -0.05 else '✗'}")
    print(f"    C1b: high-error coverage ≥ nominal−5pp (50%: Δ={hi_deficit_50:+.3f}): "
          f"{'✓' if hi_deficit_50 >= -0.05 else '✗'}")
    print(f"    C1c: boundary 90% ≥ 85pp (Δ={bnd_deficit_90:+.3f}):                  "
          f"{'✓' if bnd_deficit_90 >= -0.05 else '✗'}")
    print(f"    C2:  T transfers across R (all |Δ|<10pp):                             "
          f"{'✓' if transfer_ok else '✗'}")
    print(f"    C3:  ranking invariant to scaling (max|Δρ|={max_rho_delta:.2e}):      "
          f"{'✓' if ranking_invariant else '✗'}")

    # ── Verdict ──
    if conditional_ok and transfer_ok:
        label = "CALIBRATED_INTERVALS"
        body  = (
            f"Global temperature T={T_50:.3f} (fitted at R={R_CALIB}) restores "
            f"both marginal 50% and 90% coverage and holds conditionally across "
            f"boundary/interior and high-error/low-error pixel splits. "
            f"Temperature transfers: at R≠{R_CALIB} marginal 50% coverage gaps are "
            f"{', '.join(f'R={R}: {xfer_gaps[R]:+.3f}' for R in Rs_other)}. "
            f"Model may be used for calibrated prediction intervals. "
            f"Ranking ρ is invariant to scaling (max Δ={max_rho_delta:.2e})."
        )
    else:
        label = "RANKING-CALIBRATED_ABSTENTION"
        reasons = []
        if not conditional_ok:
            reasons.append(
                f"Conditional coverage fails: boundary Δ50%={bnd_deficit_50:+.3f}, "
                f"high-error Δ50%={hi_deficit_50:+.3f} (threshold −0.05pp). "
                f"A global T fixes marginal coverage but leaves structured subsets under-covered. "
                f"The uncertainty field is not exchangeable across gradient/error regimes."
            )
        if not transfer_ok:
            reasons.append(
                f"Temperature does not transfer: "
                f"{', '.join(f'R={R}: Δ50%={xfer_gaps[R]:+.3f}' for R in Rs_other)}. "
                f"T is regime-specific; recalibration is needed per-R."
            )
        reasons.append(
            f"The Spearman ρ (raw={check3[R_CALIB]['rho_raw']:.4f}) is preserved exactly "
            f"by temperature scaling (rank-order is monotone-invariant). "
            f"ρ>0 confirms the model knows WHERE uncertainty is high — but the intervals "
            f"are not reliable enough to publish as confidence bounds."
        )
        body = " ".join(reasons)

    print(f"\n  VERDICT: {label}")
    for line in textwrap.wrap(body, width=106):
        print(f"    {line}")
    print("═" * 110 + "\n")
    return f"{label} — {body}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    print(f"\nDevice: {device}   R_CALIB={R_CALIB}   TRAIN_EPOCHS={TRAIN_EPOCHS}   N_MC={N_MC}")

    # Load the same held-out maps used in run_calib_uncertainty.py
    print("\nLoading test maps …")
    test_maps = load_test_maps()
    print(f"  {len(test_maps)} maps")

    # ── Train models at all Rs ─────────────────────────────────────────────────
    per_map_by_R: dict[int, list[dict]] = {}
    for R in ACCEL_SWEEP:
        print(f"\n{'─'*60}\nR = {R}\n{'─'*60}")
        per_map_by_R[R] = []
        for tmap in test_maps:
            result = run_one(tmap, R, device)
            per_map_by_R[R].append(result)

    # ── Fit T at R_CALIB ──────────────────────────────────────────────────────
    maps_calib = per_map_by_R[R_CALIB]
    all_std_calib = np.concatenate([m["T2_std"][m["fg"]]  for m in maps_calib])
    all_err_calib = np.concatenate([m["abs_err"][m["fg"]] for m in maps_calib])
    fg_all = np.ones(len(all_std_calib), dtype=bool)  # all already fg-filtered

    T_50 = float(np.quantile(all_err_calib / (all_std_calib + 1e-8), 0.50) / Z_50)
    T_90 = float(np.quantile(all_err_calib / (all_std_calib + 1e-8), 0.90) / Z_90)

    print(f"\nTemperature scaling (fitted on R={R_CALIB} held-out set):")
    print(f"  T_50 = {T_50:.4f}  (targeting 50% marginal coverage)")
    print(f"  T_90 = {T_90:.4f}  (targeting 90% marginal coverage)")
    print(f"  |T_50 − T_90| / T_50 = {abs(T_50 - T_90)/T_50:.4f}  "
          f"({'consistent (Gaussian-like)' if abs(T_50-T_90)/T_50 < 0.30 else 'inconsistent — non-Gaussian tails'})")

    # ── CHECK 1 ───────────────────────────────────────────────────────────────
    check1 = check1_conditional(maps_calib, T_50, T_90)

    # ── CHECK 2 ───────────────────────────────────────────────────────────────
    check2 = check2_transfer(per_map_by_R, T_50)

    # ── CHECK 3 ───────────────────────────────────────────────────────────────
    check3 = check3_ranking(per_map_by_R, T_50)

    # ── Figures ───────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    save_figures(check1, check2, check3, T_50, T_90)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = print_verdict(check1, check2, check3, T_50, T_90)

    # ── JSON ──────────────────────────────────────────────────────────────────
    def _f(v):
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)):    return int(v)
        if isinstance(v, dict):                 return {str(k): _f(w) for k, w in v.items()}
        return v

    out = dict(
        T_50=T_50, T_90=T_90, R_CALIB=R_CALIB,
        check1=_f(check1), check2=_f(check2), check3=_f(check3),
        verdict=verdict,
    )
    with open(RESULTS_DIR / "stress_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → stress_calibration.json")
    print(f"\nFinal verdict: {verdict.split(' — ')[0]}")


if __name__ == "__main__":
    main()
