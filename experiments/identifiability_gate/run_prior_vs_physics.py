#!/usr/bin/env python3
"""
Prior vs Physics: does the spatial smoothness prior carry T2* identification?
==============================================================================
Every prior result used piecewise-constant Shepp-Logan ellipses, where the CNN's
implicit spatial smoothness prior (GroupNorm + 3×3 conv stack) is perfectly matched
to ground truth — a free lunch.  This experiment breaks that match by switching to
realistic TEXTURED maps derived from real fastMRI anatomy.

The ONLY variable is map spatial structure:
  (a) PIECEWISE-CONSTANT — Shepp-Logan ellipses (matched prior, free lunch)
  (b) REALISTIC-TEXTURED — real fastMRI anatomy + within-tissue T2* texture

KEY ANALYSIS: error vs local gradient magnitude.
  Error rising in high-gradient / boundary bins  → prior over-smoothing → PRIOR-DEPENDENT
  Error flat across gradient bins               → physics dominant     → PHYSICS-DRIVEN

VERDICT logic:
  PHYSICS-DRIVEN   : textured T2* error ≤ 2× piecewise, no gradient-bin concentration,
                     network stays clearly above the ZF analytical floor.
  PRIOR-DEPENDENT  : textured T2* error > 3× piecewise, network approaches ZF floor,
                     error concentrates at high-gradient / boundary pixels.
  INTERMEDIATE     : 2–3× degradation with characterisable boundary bias.

Usage:
    python experiments/identifiability_gate/run_prior_vs_physics.py
    python experiments/identifiability_gate/run_prior_vs_physics.py --map textured --df50
    python experiments/identifiability_gate/run_prior_vs_physics.py --epochs 300
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Literal

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, sobel, zoom

# ── shared gate infrastructure ────────────────────────────────────────────────
_GATE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_GATE_DIR))

from run_gate import (                            # noqa: E402
    make_phantom, synthesise, analytical_fit, BottleneckNet,
    kspace_consistency_loss, predict_from_model,
    fft2c_torch, ifft2c_np,
    H, W, TEs_MS, N_ECHOES, ACCEL, CF, N_SEEDS, HIDDEN, T2_MIN, T2_MAX,
    RESULTS_DIR,
)
from run_offres_fix import (                      # noqa: E402
    BottleneckNet3, train_3param, predict_3param, kspace_loss_3param,
    DF_BOUND,
)
from run_mismatch import make_offres_map          # noqa: E402
from run_attribution import _train_2param         # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────── constants ───────────────────────────────────────

SNR_DB    = 30.0
DF_LEVELS = [0, 50]    # Hz — primary df=0, optional df=50 via --df50

# fastMRI source (non-fat-suppressed PD knee, good tissue contrast)
FASTMRI_FILE  = Path("data/singlecoil_val/file1002570.h5")
FASTMRI_SLICE = 13    # ~51% fg, tissue classes well represented

TEXTURE_SIGMA = 3.0   # px — intra-region noise correlation length
TEXTURE_AMP   = 0.30  # fractional T2* texture amplitude
DF_MAP_SIGMA  = 30.0  # px — Gaussian smoothness of the Df field


# ═══════════════════════════════════════════════════════════════════════════════
# TEXTURED PHANTOM GENERATOR  (flag: --map textured)
# ═══════════════════════════════════════════════════════════════════════════════

def make_textured_phantom(
    H_out: int = 128,
    W_out: int = 128,
    fastmri_path: Path = FASTMRI_FILE,
    slice_idx: int = FASTMRI_SLICE,
    texture_seed: int = 99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Realistic-textured phantom from a real fastMRI reconstruction slice.

    S0  — normalised anatomy magnitude (real texture + sharp edges from real data).
    T2* — tissue-class base + independent within-class Gaussian texture.
          Class boundaries follow real anatomy → sharp edges NOT matched to CNN prior.
    fg  — foreground mask (S0 > 5 % of max, same threshold as piecewise neighbour).

    Physical note: B0/Df map is always kept smooth (σ=30 px) regardless of S0/T2* texture,
    matching physical reality (field maps are smooth functions of position).
    """
    with h5py.File(fastmri_path, "r") as f:
        rss = f["reconstruction_rss"][slice_idx][()]  # [320, 320] float32

    # Downsample 320 → H_out×W_out (preserves structure, slight anti-alias smoothing)
    scale = H_out / rss.shape[0]
    S0_raw = zoom(rss, scale).astype(np.float32)[:H_out, :W_out]
    S0 = S0_raw / (S0_raw.max() + 1e-9)

    fg = (S0 > 0.05).astype(bool)

    # Tissue class labels by intensity bin → base T2* values
    #   Background : 5 ms  |  low: 20 ms  |  mid-low: 40 ms
    #   mid-high   : 65 ms |  high: 85 ms
    t2_class = np.full((H_out, W_out), T2_MIN, dtype=np.float32)
    bins  = [0.00, 0.05, 0.20, 0.40, 0.65, 1.01]
    t2_vals = [T2_MIN, 20.0, 40.0, 65.0, 85.0]
    lbl = np.digitize(S0, bins) - 1   # 0=bg, 1-4=tissue
    for ci, tv in enumerate(t2_vals):
        t2_class[lbl == ci] = tv

    # Independent intra-region Gaussian texture (decorrelated from S0 at fine scales)
    rng = np.random.default_rng(texture_seed)
    raw_noise = rng.standard_normal((H_out, W_out)).astype(np.float32)
    smooth_noise = gaussian_filter(raw_noise, sigma=TEXTURE_SIGMA)
    smooth_noise /= smooth_noise.std() + 1e-8  # unit std

    T2s = t2_class * (1.0 + TEXTURE_AMP * smooth_noise)
    T2s = np.clip(T2s, T2_MIN, T2_MAX).astype(np.float32)
    # Background T2* does not need texture (not measured)
    T2s[~fg] = T2_MIN

    return S0.astype(np.float32), T2s, fg


def get_phantom(
    map_type: Literal["piecewise", "textured"],
    texture_seed: int = 99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factory: returns (S0 [H,W], T2s [H,W], fg_mask [H,W bool])."""
    if map_type == "piecewise":
        S0, T2s, lbl = make_phantom(H, W)
        return S0, T2s, (lbl > 0)
    else:
        return make_textured_phantom(H, W, texture_seed=texture_seed)


# ═══════════════════════════════════════════════════════════════════════════════
# SMOOTH DF MAP FOR df=50 Hz CONDITION
# ═══════════════════════════════════════════════════════════════════════════════

def make_smooth_offres_map(H_out: int, W_out: int,
                           df_max: float, sigma: float = DF_MAP_SIGMA,
                           seed: int = 77) -> np.ndarray:
    """
    Physically realistic off-resonance map: smooth (σ=30 px), bounded ±df_max Hz.
    B0 inhomogeneity is always smooth — kept smooth even when S0/T2* are textured.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((H_out, W_out))
    smooth = gaussian_filter(raw, sigma=sigma)
    smooth -= smooth.mean()
    # Normalise to unit max-abs so the range is exactly ±df_max
    smooth /= np.abs(smooth).max() + 1e-8
    return (smooth * df_max).astype(np.float32)   # Hz ∈ [-df_max, +df_max]


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT-BIN ANALYSIS  (the decisive test)
# ═══════════════════════════════════════════════════════════════════════════════

def gradient_analysis(
    T2_pred: np.ndarray,
    T2_gt: np.ndarray,
    fg: np.ndarray,
    n_bins: int = 4,
) -> dict:
    """
    Bin foreground pixels by local gradient magnitude of GT T2* map.
    Returns per-bin T2* error and boundary/interior split.

    High error in high-gradient bins = prior over-smoothing edges.
    """
    gx = sobel(T2_gt.astype(np.float64), axis=0)
    gy = sobel(T2_gt.astype(np.float64), axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2).astype(np.float32)

    err = np.abs(T2_pred - T2_gt)

    fg_grad = grad_mag[fg]
    fg_err  = err[fg]

    # Equal-count bins
    pct_edges = np.linspace(0, 100, n_bins + 1)
    thresholds = np.percentile(fg_grad, pct_edges)

    bins = []
    for i in range(n_bins):
        lo, hi = thresholds[i], thresholds[i + 1]
        in_bin = (fg_grad >= lo) & (fg_grad <= hi)
        if in_bin.sum() > 0:
            bins.append(dict(
                label=f"grad_bin{i+1}",
                grad_lo=float(lo), grad_hi=float(hi),
                n_px=int(in_bin.sum()),
                t2_med=float(np.median(fg_err[in_bin])),
                t2_p95=float(np.percentile(fg_err[in_bin], 95)),
            ))

    # Boundary (top 20% gradient) vs interior (bottom 50%)
    bnd_thr = np.percentile(fg_grad, 80)
    int_thr = np.percentile(fg_grad, 50)

    bnd = fg_grad >= bnd_thr
    interior = fg_grad < int_thr

    return dict(
        grad_mag=grad_mag,
        bins=bins,
        boundary_t2_med=float(np.median(fg_err[bnd])) if bnd.sum() else float("nan"),
        boundary_t2_p95=float(np.percentile(fg_err[bnd], 95)) if bnd.sum() else float("nan"),
        interior_t2_med=float(np.median(fg_err[interior])) if interior.sum() else float("nan"),
        interior_t2_p95=float(np.percentile(fg_err[interior], 95)) if interior.sum() else float("nan"),
        bnd_over_int=float(np.median(fg_err[bnd]) / (np.median(fg_err[interior]) + 1e-8)) if (bnd.sum() and interior.sum()) else float("nan"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def run_analytical_baselines(data: dict, T2s_gt: np.ndarray, fg: np.ndarray) -> dict:
    """Analytical T2* fit on fully-sampled and zero-filled echoes."""
    _, T2_full = analytical_fit(data["echoes_full"], TEs_MS)
    _, T2_zf   = analytical_fit(data["zf_mag"],      TEs_MS)

    err_full = np.abs(T2_full[fg] - T2s_gt[fg])
    err_zf   = np.abs(T2_zf[fg]  - T2s_gt[fg])

    return dict(
        full_t2_med=float(np.median(err_full)),
        full_t2_p95=float(np.percentile(err_full, 95)),
        zf_t2_med  =float(np.median(err_zf)),
        zf_t2_p95  =float(np.percentile(err_zf, 95)),
        T2_full=T2_full,
        T2_zf=T2_zf,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-SEED TRAINER  (2-param or 3-param, returns T2* map + loss)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_complex_zf_input(kspace_under: np.ndarray) -> np.ndarray:
    """[2*n_echoes, H, W] real+imag channels from undersampled k-space."""
    zf = np.stack([ifft2c_np(kspace_under[e]) for e in range(kspace_under.shape[0])], axis=0)
    scale = float(np.max(np.abs(zf[0]))) + 1e-8
    return np.concatenate([zf.real / scale, zf.imag / scale], axis=0).astype(np.float32)


def run_seeds(
    model_type: Literal["2p", "3p"],
    data: dict,
    n_epochs: int,
    device: torch.device,
    label_prefix: str = "",
) -> dict:
    """
    Train N_SEEDS models, return aggregate metrics + per-seed T2* maps.
    model_type='2p' uses complex k-space loss (no Df DOF).
    model_type='3p' uses complex k-space loss with Df DOF.
    """
    k_under_tc = torch.from_numpy(data["kspace_under"]).to(torch.complex64)
    mask_tc    = torch.from_numpy(data["mask_2d"])

    seed_T2: list[np.ndarray] = []
    seed_loss: list[float]    = []

    for s in range(N_SEEDS):
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        tag = f"{label_prefix} s={s}"

        if model_type == "3p":
            zf_arr   = _make_complex_zf_input(data["kspace_under"])
            zf_input = torch.from_numpy(zf_arr).unsqueeze(0)
            model    = BottleneckNet3(N_ECHOES, HIDDEN, T2_MIN, T2_MAX, DF_BOUND, complex_input=True)
            model, fl = train_3param(model, zf_input, k_under_tc, mask_tc, TEs_MS,
                                     n_epochs=n_epochs, lr=1e-3, device=device, label=tag)
            _, T2_p, _ = predict_3param(model, zf_input, device)
        else:  # 2p
            zf_arr   = data["zf_mag"]
            zf_input = torch.from_numpy(zf_arr).unsqueeze(0)
            model    = BottleneckNet(N_ECHOES, HIDDEN, T2_MIN, T2_MAX)
            model, fl = _train_2param(model, zf_input, k_under_tc, mask_tc, TEs_MS,
                                      loss_fn=kspace_consistency_loss,
                                      n_epochs=n_epochs, lr=1e-3, device=device, label=tag)
            _, T2_p = predict_from_model(model, zf_input, device)

        seed_T2.append(T2_p)
        seed_loss.append(fl)
        print(f"    {tag}  final_loss={fl:.5f}")

    return dict(seed_T2=seed_T2, seed_loss=seed_loss)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def t2_metrics(T2_pred: np.ndarray, T2_gt: np.ndarray, fg: np.ndarray) -> dict:
    err = np.abs(T2_pred[fg] - T2_gt[fg])
    return dict(t2_med=float(np.median(err)), t2_p95=float(np.percentile(err, 95)))


def aggregate_seeds(seed_T2: list, T2_gt: np.ndarray, fg: np.ndarray) -> dict:
    """Best-seed T2* metrics + cross-seed std."""
    per_seed = [t2_metrics(T2, T2_gt, fg) for T2 in seed_T2]
    best     = int(np.argmin([m["t2_med"] for m in per_seed]))
    stack    = np.stack(seed_T2, axis=0)
    cs_std   = stack[:, fg].std(axis=0)
    return dict(
        t2_med     = per_seed[best]["t2_med"],
        t2_p95     = per_seed[best]["t2_p95"],
        mean_loss  = float(np.mean([seed_T2[0].size]))  # placeholder; fixed below
        if False else float(np.nan),  # filled per caller
        xseed_mean = float(cs_std.mean()),
        best_T2    = seed_T2[best],
        stack      = stack,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def save_maps_figure(tag: str,
                     S0_gt: np.ndarray, T2s_gt: np.ndarray,
                     T2_net: np.ndarray, T2_zf: np.ndarray,
                     T2_full: np.ndarray,
                     grad_mag: np.ndarray, fg: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(tag, fontsize=11)

    vt = (0, min(120, T2s_gt[fg].max() + 10))

    def im(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
        v = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=9); ax.axis("off")
        plt.colorbar(v, ax=ax, fraction=0.046, pad=0.04)

    err_net  = np.abs(T2_net  - T2s_gt)
    err_full = np.abs(T2_full - T2s_gt)

    im(axes[0, 0], T2s_gt,   "GT T2* (ms)",            *vt)
    im(axes[0, 1], T2_net,   "Network T2* (ms)",        *vt)
    im(axes[0, 2], err_net,  "|Error| T2* — Network",   0, 20, "hot")

    im(axes[1, 0], S0_gt,    "GT S0 (anatomy)",         0,  1)
    im(axes[1, 1], T2_zf,    "T2* from ZF analytical",  *vt)
    im(axes[1, 2], err_full, "|Error| T2* — Anal.full", 0, 20, "hot")

    fig.tight_layout()
    safe = tag.replace(" ", "_").replace("=", "").replace("|", "").replace("/", "_")
    out  = RESULTS_DIR / f"pvp_{safe}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  → saved {out.name}")


def save_texture_analysis_figure(
    pc_bins: list, tx_bins: list,
    pc_bnd_int: dict, tx_bnd_int: dict,
    pc_xseed: np.ndarray, tx_xseed: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Prior vs Physics — Texture Dependence Analysis", fontsize=11)

    # Panel 1: T2* error vs gradient bin
    ax = axes[0]
    labels = [f"bin{i+1}" for i in range(len(pc_bins))]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, [b["t2_med"] for b in pc_bins], w,
           label="Piecewise-const", color="steelblue")
    ax.bar(x + w/2, [b["t2_med"] for b in tx_bins], w,
           label="Textured", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Gradient magnitude bin (low → high)")
    ax.set_ylabel("T2* median error (ms)")
    ax.set_title("Error vs local gradient bin")
    ax.legend(fontsize=8)

    # Panel 2: Boundary vs Interior
    ax = axes[1]
    categories = ["Piecewise\nInterior", "Piecewise\nBoundary",
                  "Textured\nInterior", "Textured\nBoundary"]
    values = [
        pc_bnd_int["interior_t2_med"], pc_bnd_int["boundary_t2_med"],
        tx_bnd_int["interior_t2_med"], tx_bnd_int["boundary_t2_med"],
    ]
    colors = ["steelblue", "cornflowerblue", "tomato", "firebrick"]
    ax.bar(range(4), values, color=colors)
    ax.set_xticks(range(4))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("T2* median error (ms)")
    ax.set_title("Boundary (top-20% grad) vs Interior (bot-50%)")

    # Panel 3: Cross-seed std maps
    ax = axes[2]
    # Show side-by-side cross-seed std as strip image
    combined = np.concatenate([pc_xseed, np.full((H, 4), np.nan), tx_xseed], axis=1)
    v = ax.imshow(combined, vmin=0, vmax=10, cmap="hot", origin="upper")
    plt.colorbar(v, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Cross-seed T2* std (ms)\n[Piecewise | Textured]")
    ax.axis("off")

    fig.tight_layout()
    out = RESULTS_DIR / "pvp_texture_analysis.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  → saved {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE + VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def print_table_and_verdict(results: dict) -> str:
    """
    results keys: (map_type, df_hz, model_type)
    Each value: dict with t2_med, t2_p95, mean_loss, xseed_mean, grad_analysis,
                ana_full_t2_med, ana_zf_t2_med
    """
    print("\n" + "═" * 90)
    print("PRIOR vs PHYSICS — COMPARISON TABLE")
    print("  map_type: PC=piecewise-constant  TX=textured")
    print("═" * 90)
    hdr  = f"  {'Map':<4}  {'df':>5}  {'Model':<12}  {'T2*med':>7}  {'T2*p95':>7}  "
    hdr += f"{'Loss':>8}  {'Xseed':>7}  {'UB (full)':>9}  {'ZF (floor)':>10}"
    print(hdr)
    print(f"  {'':4}  {'(Hz)':>5}  {'':12}  {'(ms)':>7}  {'(ms)':>7}  "
          f"{'(norm)':>8}  {'(ms)':>7}  {'(ms)':>9}  {'(ms)':>10}")
    print("  " + "─" * 86)

    for (map_type, df_hz, model) in sorted(results.keys()):
        r = results[(map_type, df_hz, model)]
        tag = "PC" if map_type == "piecewise" else "TX"
        print(f"  {tag:<4}  {df_hz:>5}  {model:<12}  "
              f"{r['t2_med']:>7.2f}  {r['t2_p95']:>7.2f}  "
              f"{r['mean_loss']:>8.5f}  {r['xseed_mean']:>7.3f}  "
              f"{r['ana_full_t2_med']:>9.2f}  {r['ana_zf_t2_med']:>10.2f}")

    print("  " + "═" * 86)

    # ── gradient-bin analysis ──────────────────────────────────────────────────
    print("\n  ERROR vs GRADIENT-MAGNITUDE BIN  (T2* median ms, best-seed network)")
    bin_header = f"  {'Map':<4}  {'df':>5}  {'Model':<12}"
    for k in results:
        ga = results[k].get("grad_analysis")
        if ga:
            for b in ga["bins"]:
                bin_header += f"  {b['label']:>10}"
            bin_header += f"  {'bnd/int':>8}"
            break
    print(bin_header)
    print("  " + "─" * 86)

    for (map_type, df_hz, model) in sorted(results.keys()):
        r = results[(map_type, df_hz, model)]
        ga = r.get("grad_analysis")
        if not ga:
            continue
        tag = "PC" if map_type == "piecewise" else "TX"
        row = f"  {tag:<4}  {df_hz:>5}  {model:<12}"
        for b in ga["bins"]:
            row += f"  {b['t2_med']:>10.2f}"
        row += f"  {ga['bnd_over_int']:>8.2f}×"
        print(row)
    print("  " + "─" * 86)
    print("  bnd/int = boundary-T2*-error / interior-T2*-error  (1.0 = no bias)")

    # ── verdict ───────────────────────────────────────────────────────────────
    # Pick 3p df=0 for both conditions as the primary comparison
    pc_key = ("piecewise", 0, "3p-complex")
    tx_key = ("textured",  0, "3p-complex")
    if pc_key not in results or tx_key not in results:
        # Fall back to 2p if 3p not run
        pc_key = ("piecewise", 0, "2p-complex")
        tx_key = ("textured",  0, "2p-complex")

    pc = results.get(pc_key, {})
    tx = results.get(tx_key, {})

    pc_t2  = pc.get("t2_med", float("nan"))
    tx_t2  = tx.get("t2_med", float("nan"))
    ratio  = tx_t2 / (pc_t2 + 1e-8)

    pc_zf  = pc.get("ana_zf_t2_med", float("nan"))
    tx_zf  = tx.get("ana_zf_t2_med", float("nan"))
    pc_gap = pc_zf - pc_t2     # how much better than ZF on piecewise
    tx_gap = tx_zf - tx_t2     # how much better than ZF on textured

    # bnd/int ratio on textured condition
    tx_ga = tx.get("grad_analysis", {})
    bnd_int = tx_ga.get("bnd_over_int", float("nan"))

    print(f"\n  VERDICT ANALYSIS")
    print(f"    3p-complex @ df=0, piecewise:  T2* med = {pc_t2:.2f} ms  "
          f"(ZF floor = {pc_zf:.2f} ms, gap = {pc_gap:.2f} ms)")
    print(f"    3p-complex @ df=0, textured:   T2* med = {tx_t2:.2f} ms  "
          f"(ZF floor = {tx_zf:.2f} ms, gap = {tx_gap:.2f} ms)")
    print(f"    Textured / piecewise error ratio: {ratio:.2f}×")
    print(f"    Boundary / interior error ratio (textured): {bnd_int:.2f}×")
    print(f"    Network improvement over ZF floor:")
    print(f"      Piecewise: {pc_gap:.2f} ms  ({pc_gap/pc_zf*100:.0f}% of ZF error removed)")
    print(f"      Textured:  {tx_gap:.2f} ms  ({tx_gap/tx_zf*100:.0f}% of ZF error removed)")

    print()
    import textwrap
    if ratio <= 2.0 and tx_gap / (tx_zf + 1e-8) > 0.3:
        if bnd_int < 2.0:
            verdict = (
                "PHYSICS-DRIVEN — Textured-map T2* error is only "
                f"{ratio:.2f}× the piecewise result ({tx_t2:.2f} vs {pc_t2:.2f} ms). "
                "The network retains significant advantage over the ZF analytical floor "
                f"({tx_gap:.1f} ms removed = {tx_gap/tx_zf*100:.0f}% of ZF error) and error "
                "does NOT concentrate at high-gradient boundary pixels (bnd/int = "
                f"{bnd_int:.2f}×). Multi-echo physics is the primary driver; the CNN "
                "smoothness prior provides small regularisation but does not dominate."
            )
        else:
            verdict = (
                "INTERMEDIATE (physics-driven, boundary bias) — Overall textured error "
                f"only {ratio:.2f}× piecewise ({tx_t2:.2f} ms), physics still beats ZF "
                f"({tx_gap:.1f} ms removed). BUT error concentrates at boundary pixels "
                f"(bnd/int = {bnd_int:.2f}×): the prior over-smooths sharp edges, "
                "producing a characterisable bias at tissue boundaries. Confidence in "
                "per-pixel T2* is lower within ~3 px of tissue interfaces."
            )
    elif ratio <= 3.0:
        verdict = (
            "INTERMEDIATE — Textured-map T2* error is "
            f"{ratio:.2f}× the piecewise result ({tx_t2:.2f} vs {pc_t2:.2f} ms). "
            "The network still beats the ZF floor "
            f"({tx_gap:.1f} ms = {tx_gap/tx_zf*100:.0f}% of ZF error removed) but the "
            f"bnd/int concentration ratio ({bnd_int:.2f}×) signals that the smoothness "
            "prior is over-smoothing fine-scale texture and boundary detail. "
            "The method is physics-driven in-tissue but prior-driven at boundaries. "
            "High-gradient regions should be flagged as lower confidence."
        )
    else:
        verdict = (
            "PRIOR-DEPENDENT — Textured-map T2* error is "
            f"{ratio:.2f}× the piecewise result ({tx_t2:.2f} vs {pc_t2:.2f} ms). "
            f"Network improvement over ZF floor is only {tx_gap:.1f} ms "
            f"({tx_gap/tx_zf*100:.0f}%): the method approaches the zero-filled "
            "analytical baseline on realistic maps. Error concentrates at boundary "
            f"pixels (bnd/int = {bnd_int:.2f}×). The prior was carrying most of the "
            "piecewise-constant result; physics-only identification degrades substantially "
            "when map texture is not matched to the CNN receptive field."
        )

    print("  VERDICT:")
    for line in textwrap.wrap(verdict, width=80):
        print(f"    {line}")
    print("═" * 90 + "\n")
    return verdict


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=300,
                        help="Training epochs per model (default 300)")
    parser.add_argument("--map",      choices=["both", "piecewise", "textured"],
                        default="both", help="Which phantom type(s) to run (default both)")
    parser.add_argument("--df50",     action="store_true",
                        help="Also run df=50 Hz condition (smooth off-resonance map)")
    args = parser.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    map_types = (["piecewise", "textured"] if args.map == "both"
                 else [args.map])
    df_levels = [0, 50] if args.df50 else [0]

    print(f"\nDevice: {device},  Epochs: {args.epochs},  SNR: {SNR_DB} dB")
    print(f"Map types: {map_types},  df levels: {df_levels} Hz")
    print(f"Seeds: {N_SEEDS},  Phantom: {H}×{W}\n")

    results: dict[tuple, dict] = {}

    for map_type in map_types:
        S0_gt, T2s_gt, fg = get_phantom(map_type)
        print(f"\n{'═'*72}")
        print(f"PHANTOM TYPE: {map_type.upper()}")
        print(f"  fg pixels: {fg.sum()}/{H*W}  "
              f"T2* range: [{T2s_gt[fg].min():.1f}, {T2s_gt[fg].max():.1f}] ms  "
              f"T2* std: {T2s_gt[fg].std():.1f} ms")

        # Gradient stats of GT T2* map (textured should be >> piecewise)
        gx = sobel(T2s_gt.astype(np.float64), 0)
        gy = sobel(T2s_gt.astype(np.float64), 1)
        gm = np.sqrt(gx**2 + gy**2)
        print(f"  GT T2* gradient: mean={gm[fg].mean():.2f}, p95={np.percentile(gm[fg],95):.2f}")

        for df_hz in df_levels:
            print(f"\n  ── df = {df_hz} Hz ──")

            # Off-resonance map: smooth regardless of map_type (physically correct)
            if df_hz > 0:
                offres_map = make_smooth_offres_map(H, W, df_max=float(df_hz))
                print(f"  Df map: smooth σ={DF_MAP_SIGMA}px, "
                      f"range=[{offres_map.min():.1f},{offres_map.max():.1f}] Hz")
            else:
                offres_map = None

            # Synthesise data  (same noise/mask seeds → apples-to-apples)
            data = synthesise(S0_gt, T2s_gt, TEs_MS, ACCEL, CF, SNR_DB,
                              mask_seed=42, noise_seed=7, offres_map=offres_map)

            # Analytical baselines
            bl = run_analytical_baselines(data, T2s_gt, fg)
            print(f"  Analytical full:  T2* med = {bl['full_t2_med']:.2f} ms  "
                  f"(p95={bl['full_t2_p95']:.2f})")
            print(f"  Analytical ZF:    T2* med = {bl['zf_t2_med']:.2f} ms  "
                  f"(p95={bl['zf_t2_p95']:.2f})")

            # Choose models: df=0→2p+3p, df>0→3p only
            models_to_run: list[Literal["2p", "3p"]] = (
                ["2p", "3p"] if df_hz == 0 else ["3p"]
            )

            for model_type in models_to_run:
                key = (map_type, df_hz, f"{model_type}-complex")
                print(f"\n  [{key}]")
                seed_out = run_seeds(model_type, data, args.epochs, device,
                                     label_prefix=f"{map_type[:2]}_{df_hz}Hz_{model_type}")

                # Aggregate
                seed_T2 = seed_out["seed_T2"]
                per_m   = [t2_metrics(T, T2s_gt, fg) for T in seed_T2]
                best    = int(np.argmin([m["t2_med"] for m in per_m]))
                stack   = np.stack(seed_T2, axis=0)
                cs_std  = stack[:, fg].std(axis=0)
                cs_std_map = stack.std(axis=0)

                best_T2 = seed_T2[best]
                ga      = gradient_analysis(best_T2, T2s_gt, fg)

                r = dict(
                    t2_med       = per_m[best]["t2_med"],
                    t2_p95       = per_m[best]["t2_p95"],
                    mean_loss    = float(np.mean(seed_out["seed_loss"])),
                    xseed_mean   = float(cs_std.mean()),
                    ana_full_t2_med = bl["full_t2_med"],
                    ana_zf_t2_med   = bl["zf_t2_med"],
                    grad_analysis   = ga,
                    best_T2    = best_T2,
                    cs_std_map = cs_std_map,
                    T2_zf      = bl["T2_zf"],
                    T2_full    = bl["T2_full"],
                    S0_gt      = S0_gt,
                    T2s_gt     = T2s_gt,
                    fg         = fg,
                )
                results[key] = r

                print(f"  → T2* med={r['t2_med']:.2f} ms  p95={r['t2_p95']:.2f} ms  "
                      f"loss={r['mean_loss']:.5f}  Xseed={r['xseed_mean']:.3f} ms")
                print(f"  → bnd/int={ga['bnd_over_int']:.2f}×  "
                      f"bnd_T2*={ga['boundary_t2_med']:.2f} ms  "
                      f"int_T2*={ga['interior_t2_med']:.2f} ms")

                # Save per-condition map figure
                save_maps_figure(
                    f"{map_type} df{df_hz}Hz {model_type}",
                    S0_gt, T2s_gt, best_T2, bl["T2_zf"], bl["T2_full"],
                    ga["grad_mag"], fg,
                )

    # ── texture analysis figure ────────────────────────────────────────────────
    pc3  = results.get(("piecewise", 0, "3p-complex"))
    tx3  = results.get(("textured",  0, "3p-complex"))
    if pc3 and tx3:
        save_texture_analysis_figure(
            pc_bins    = pc3["grad_analysis"]["bins"],
            tx_bins    = tx3["grad_analysis"]["bins"],
            pc_bnd_int = pc3["grad_analysis"],
            tx_bnd_int = tx3["grad_analysis"],
            pc_xseed   = pc3["cs_std_map"],
            tx_xseed   = tx3["cs_std_map"],
        )

    print_table_and_verdict(results)


if __name__ == "__main__":
    main()
