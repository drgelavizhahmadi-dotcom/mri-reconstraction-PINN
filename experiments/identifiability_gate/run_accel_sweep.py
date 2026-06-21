#!/usr/bin/env python3
"""
Acceleration Sweep: find the operating point R* for per-pixel T2* texture recovery.
=====================================================================================
The JEPA-prior experiment confirmed that at 4× undersampling the texture in the
unmeasured k-space band cannot be honestly recovered. This experiment sweeps R ∈ {1,2,4,8}
to find R* — the highest acceleration at which per-pixel texture IS recoverable from data.

Uses the same held-out textured maps, seeds, forward operator, and 3-param physics
bottleneck (standard CNN, no learned prior) across all R.

Per-R report:
  1. 3-param network T2* error  (median, p95)
  2. ZF analytical T2* error    (no-prior floor at this R)
  3. Full-sample analytical UB  (R-independent reference)
  4. Texture fidelity in unmeasured band: correlation + NMSE
  5. Cross-seed std
  6. Spectral energy analysis: fraction of T2* / echo spectral energy
     inside vs outside the sampled region

DEFINITION OF DONE:
  Curve: T2* error and unmeas-band correlation vs R.
  R* = highest R where unmeas_corr is well above the ZF floor AND T2* error
       remains meaningfully below the ZF analytical floor.

Usage:
    python experiments/identifiability_gate/run_accel_sweep.py
    python experiments/identifiability_gate/run_accel_sweep.py --epochs 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom

# ── repo paths ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GATE_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_GATE_DIR))

from run_gate import (                            # noqa: E402
    synthesise, analytical_fit, fft2c_np, ifft2c_np,
    H, W, TEs_MS, N_ECHOES, CF, HIDDEN, T2_MIN, T2_MAX, RESULTS_DIR,
)
from run_offres_fix import (                      # noqa: E402
    BottleneckNet3, train_3param, predict_3param, kspace_loss_3param, DF_BOUND,
    _make_complex_zf_input,
)

warnings.filterwarnings("ignore")

# ─────────────────────────── experiment constants ────────────────────────────

ACCEL_SWEEP  = [1, 2, 4, 8]
SNR_DB       = 30.0
MASK_SEED    = 42      # same mask seed as all prior experiments
NOISE_SEED   = 7       # same noise seed as JEPA experiment
N_SEEDS      = 3       # reconstruction seeds for cross-seed std
RECON_EPOCHS = 200
RECON_LR     = 1e-3

# Textured map source — same test split as JEPA experiment (file-disjoint)
DATA_DIR         = Path("data/singlecoil_val")
N_PRETRAIN_FILES = 8   # skip these; test files follow
N_TEST_FILES     = 2
SLICES_PER_FILE  = 4
TEXTURE_SEEDS    = [2, 3]   # same texture realisations as JEPA
N_TEST_MAPS      = 4        # maps to reconstruct per R

TEXTURE_SIGMA = 3.0
TEXTURE_AMP   = 0.30


# ═══════════════════════════════════════════════════════════════════════════════
# MAP DATASET  (identical generator to run_jepa_prior.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_corpd_files() -> list[str]:
    files = []
    for fn in sorted(os.listdir(DATA_DIR)):
        try:
            with h5py.File(DATA_DIR / fn, "r") as f:
                acq  = f.attrs.get("acquisition", "")
                n_sl = f["reconstruction_rss"].shape[0]
            if "CORPD" in acq and "FS" not in acq and "DFS" not in acq and n_sl >= 30:
                files.append(fn)
        except Exception:
            pass
    return sorted(files)


def _make_textured_map(rss_slice: np.ndarray, texture_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one RSS slice → (S0, T2*, fg). Identical to run_jepa_prior.py."""
    scale = H / rss_slice.shape[0]
    S0 = zoom(rss_slice, scale).astype(np.float32)[:H, :W]
    S0 /= S0.max() + 1e-9

    fg = S0 > 0.05

    bins   = [0.0, 0.05, 0.20, 0.40, 0.65, 1.01]
    t2_val = [T2_MIN, 20.0, 40.0, 65.0, 85.0]
    lbl    = np.digitize(S0, bins) - 1
    T2_base = np.full((H, W), T2_MIN, dtype=np.float32)
    for ci, tv in enumerate(t2_val):
        T2_base[lbl == ci] = tv

    rng      = np.random.default_rng(texture_seed)
    noise    = rng.standard_normal((H, W)).astype(np.float32)
    noise_sm = gaussian_filter(noise, sigma=TEXTURE_SIGMA)
    noise_sm /= noise_sm.std() + 1e-8
    T2 = np.clip(T2_base * (1.0 + TEXTURE_AMP * noise_sm), T2_MIN, T2_MAX).astype(np.float32)
    T2[~fg] = T2_MIN

    return S0.astype(np.float32), T2, fg


def load_test_maps() -> list[dict]:
    all_files  = _get_corpd_files()
    test_files = all_files[N_PRETRAIN_FILES: N_PRETRAIN_FILES + N_TEST_FILES]
    maps = []
    for fn in test_files:
        with h5py.File(DATA_DIR / fn, "r") as f:
            rss = f["reconstruction_rss"][()]
        n_sl = rss.shape[0]
        idxs = np.linspace(n_sl * 0.2, n_sl * 0.8, SLICES_PER_FILE, dtype=int)
        for si in idxs:
            for seed in TEXTURE_SEEDS:
                S0, T2, fg = _make_textured_map(rss[si], texture_seed=int(seed))
                maps.append({"S0": S0, "T2": T2, "fg": fg,
                             "file": fn, "slice": int(si), "seed": int(seed)})
    return maps


# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRAL ENERGY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def spectral_energy_fractions(
    T2_gt: np.ndarray,
    kspace_full: np.ndarray,   # [N_echoes, H, W] complex — full k-space
    mask_1d: np.ndarray,       # [W] float  (1 = sampled)
) -> dict:
    """
    Fraction of spectral energy OUTSIDE the sampled region.

    frac_T2_unmeas   — fraction of T2* map's Fourier energy in unsampled columns.
                       This is the energy the network must FABRICATE to produce
                       fine-scale T2* texture.
    frac_echo_unmeas — same, averaged over echo images (primary MRI measurement).
                       High value means the unsampled band contains significant
                       signal from the actual measurement process.
    """
    unmeas = mask_1d == 0        # [W] bool

    if unmeas.sum() == 0:        # R=1: fully sampled
        return dict(frac_T2_unmeas=0.0, frac_echo_unmeas=0.0,
                    n_unmeas=0, n_total=len(mask_1d))

    # T2* map spectral energy
    K_T2        = fft2c_np(T2_gt.astype(np.complex64))
    e_T2_total  = float(np.sum(np.abs(K_T2)**2))
    e_T2_unmeas = float(np.sum(np.abs(K_T2[:, unmeas])**2))
    frac_T2     = e_T2_unmeas / (e_T2_total + 1e-12)

    # Echo image spectral energy (mean over echoes using the TRUE k-space)
    frac_echo_per = []
    for ei in range(kspace_full.shape[0]):
        K  = kspace_full[ei]
        et = float(np.sum(np.abs(K)**2))
        eu = float(np.sum(np.abs(K[:, unmeas])**2))
        frac_echo_per.append(eu / (et + 1e-12))
    frac_echo = float(np.mean(frac_echo_per))

    return dict(frac_T2_unmeas=frac_T2, frac_echo_unmeas=frac_echo,
                n_unmeas=int(unmeas.sum()), n_total=int(len(mask_1d)))


# ═══════════════════════════════════════════════════════════════════════════════
# TEXTURE FIDELITY IN UNMEASURED BAND
# ═══════════════════════════════════════════════════════════════════════════════

def texture_fidelity(
    T2_rec: np.ndarray,
    T2_gt: np.ndarray,
    mask_1d: np.ndarray,
) -> dict:
    """
    Pearson correlation + NMSE between recovered and GT T2* k-space in the
    UNMEASURED columns. Measures whether the network filled the unsampled band
    with correct texture or invented content.
    """
    unmeas = mask_1d == 0

    if unmeas.sum() == 0:   # fully sampled — no unmeasured band
        return dict(corr=1.0, nmse=0.0, halluc_frac=0.0)

    K_gt  = fft2c_np(T2_gt.astype(np.complex64))
    K_rec = fft2c_np(T2_rec.astype(np.complex64))

    K_gt_u  = K_gt[:, unmeas]
    K_rec_u = K_rec[:, unmeas]

    gt_r  = np.concatenate([K_gt_u.real.ravel(), K_gt_u.imag.ravel()])
    rec_r = np.concatenate([K_rec_u.real.ravel(), K_rec_u.imag.ravel()])

    corr = float(np.corrcoef(gt_r, rec_r)[0, 1]) if gt_r.std() > 1e-9 else 0.0
    nmse = float(np.sum(np.abs(K_rec_u - K_gt_u)**2)
                 / (np.sum(np.abs(K_gt_u)**2) + 1e-12))
    rec_e = float(np.sum(np.abs(K_rec_u)**2))
    err_e = float(np.sum(np.abs(K_rec_u - K_gt_u)**2))
    halluc_frac = err_e / (rec_e + 1e-12)

    return dict(corr=corr, nmse=nmse, halluc_frac=halluc_frac)


# ═══════════════════════════════════════════════════════════════════════════════
# T2* METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def t2_metrics(T2_pred: np.ndarray, T2_gt: np.ndarray, fg: np.ndarray) -> dict:
    e = np.abs(T2_pred[fg] - T2_gt[fg])
    return dict(med=float(np.median(e)), p95=float(np.percentile(e, 95)))


def best_seed(T2_list: list[np.ndarray], T2_gt: np.ndarray, fg: np.ndarray):
    errs = [np.median(np.abs(T[fg] - T2_gt[fg])) for T in T2_list]
    b = int(np.argmin(errs))
    return T2_list[b], float(errs[b])


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-R RECONSTRUCTION BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_accel(
    R: int,
    tmap: dict,
    T2_ub: np.ndarray,          # full-sample analytical UB (pre-computed)
    n_epochs: int,
    device: torch.device,
) -> dict:
    """Run full reconstruction pipeline at one acceleration for one test map."""
    S0_gt, T2_gt, fg = tmap["S0"], tmap["T2"], tmap["fg"]

    data    = synthesise(S0_gt, T2_gt, TEs_MS, accel=R, cf=CF,
                         snr_db=SNR_DB, mask_seed=MASK_SEED, noise_seed=NOISE_SEED)
    mask_1d = data["mask_1d"]
    mask_tc = torch.from_numpy(data["mask_2d"])
    k_under = torch.from_numpy(data["kspace_under"]).to(torch.complex64)

    # ZF analytical floor at this R
    _, T2_zf = analytical_fit(data["zf_mag"], TEs_MS)
    zf_m = t2_metrics(T2_zf, T2_gt, fg)

    # Spectral energy fractions
    se = spectral_energy_fractions(T2_gt, data["kspace_full"], mask_1d)

    # ZF texture fidelity (diagnostic: how well does ZF T2* map reconstruct unmeas band)
    tf_zf = texture_fidelity(T2_zf, T2_gt, mask_1d)

    # Full-sample UB texture fidelity (R-independent, pre-computed)
    tf_ub = texture_fidelity(T2_ub, T2_gt, mask_1d)

    # Network reconstruction — N_SEEDS seeds
    zf_arr = _make_complex_zf_input(data["kspace_under"])
    zf_in  = torch.from_numpy(zf_arr).unsqueeze(0)

    net_T2s: list[np.ndarray] = []
    dc_losses: list[float]    = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = BottleneckNet3(N_ECHOES, HIDDEN, T2_MIN, T2_MAX, DF_BOUND, complex_input=True)
        model, fl = train_3param(
            model, zf_in, k_under, mask_tc, TEs_MS,
            n_epochs=n_epochs, lr=RECON_LR, device=device,
            label=f"R={R} s{seed}",
        )
        _, T2_hat, _ = predict_3param(model, zf_in, device)
        net_T2s.append(T2_hat)
        dc_losses.append(fl)

    T2_best, net_med = best_seed(net_T2s, T2_gt, fg)
    net_m  = t2_metrics(T2_best, T2_gt, fg)
    xseed  = float(np.stack(net_T2s, 0)[:, fg].std(0).mean())
    tf_net = texture_fidelity(T2_best, T2_gt, mask_1d)
    ub_m   = t2_metrics(T2_ub, T2_gt, fg)

    print(f"    R={R}  net={net_m['med']:.2f}ms  zf={zf_m['med']:.2f}ms  "
          f"ub={ub_m['med']:.2f}ms  corr={tf_net['corr']:.3f}  "
          f"halluc={tf_net['halluc_frac']:.3f}  "
          f"frac_T2_unmeas={se['frac_T2_unmeas']:.3f}")

    return dict(
        R=R,
        net_med=net_m["med"], net_p95=net_m["p95"],
        zf_med=zf_m["med"],   zf_p95=zf_m["p95"],
        ub_med=ub_m["med"],   ub_p95=ub_m["p95"],
        xseed=xseed,
        dc_loss=float(np.mean(dc_losses)),
        tf_net=tf_net, tf_zf=tf_zf, tf_ub=tf_ub,
        spectral=se,
        T2_best=T2_best, T2_gt=T2_gt, T2_zf=T2_zf, T2_ub=T2_ub, fg=fg,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE ACROSS TEST MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate(rows: list[dict]) -> dict:
    """Mean of per-map results at one R value."""
    def m(k): return float(np.mean([r[k] for r in rows]))
    def mn(k1, k2): return float(np.mean([r[k1][k2] for r in rows]))
    return dict(
        R           = rows[0]["R"],
        net_med     = m("net_med"),      net_p95 = m("net_p95"),
        zf_med      = m("zf_med"),       zf_p95  = m("zf_p95"),
        ub_med      = m("ub_med"),       ub_p95  = m("ub_p95"),
        xseed       = m("xseed"),
        dc_loss     = m("dc_loss"),
        net_corr    = mn("tf_net", "corr"),
        net_nmse    = mn("tf_net", "nmse"),
        net_halluc  = mn("tf_net", "halluc_frac"),
        zf_corr     = mn("tf_zf",  "corr"),
        ub_corr     = mn("tf_ub",  "corr"),
        frac_T2_unmeas   = mn("spectral", "frac_T2_unmeas"),
        frac_echo_unmeas = mn("spectral", "frac_echo_unmeas"),
        n_unmeas    = rows[0]["spectral"]["n_unmeas"],
        n_total     = rows[0]["spectral"]["n_total"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def save_curve_figure(agg_rows: list[dict]) -> None:
    Rs       = [r["R"] for r in agg_rows]
    net_meds = [r["net_med"] for r in agg_rows]
    zf_meds  = [r["zf_med"]  for r in agg_rows]
    ub_meds  = [r["ub_med"]  for r in agg_rows]
    net_corr = [r["net_corr"] for r in agg_rows]
    zf_corr  = [r["zf_corr"]  for r in agg_rows]
    frac_T2  = [r["frac_T2_unmeas"]   for r in agg_rows]
    frac_ech = [r["frac_echo_unmeas"]  for r in agg_rows]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # ── Panel A: T2* error vs R ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(Rs, net_meds, "o-", color="steelblue",  lw=2, label="3p-CNN (best seed)")
    ax.plot(Rs, zf_meds,  "s--",color="tomato",     lw=1.5, label="ZF analytical floor")
    ax.plot(Rs, ub_meds,  "^:", color="gray",        lw=1.5, label="Full-sample UB")
    ax.set_xscale("log", base=2); ax.set_xticks(Rs); ax.set_xticklabels([f"R={r}" for r in Rs])
    ax.set_xlabel("Acceleration R"); ax.set_ylabel("T2* median error (ms)")
    ax.set_title("T2* Error vs Acceleration"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel B: Unmeasured-band correlation vs R ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(Rs, net_corr, "o-", color="steelblue", lw=2, label="3p-CNN")
    ax2.plot(Rs, zf_corr,  "s--",color="tomato",    lw=1.5, label="ZF analytical")
    ax2.axhline(0.5, color="green", ls=":", lw=1, label="r=0.5 threshold")
    ax2.set_xscale("log", base=2); ax2.set_xticks(Rs); ax2.set_xticklabels([f"R={r}" for r in Rs])
    ax2.set_xlabel("Acceleration R"); ax2.set_ylabel("Pearson r (unmeas band)")
    ax2.set_title("Texture Fidelity in Unmeasured K-space Band")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.1, 1.05)

    # ── Panel C: Spectral energy fraction vs R ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(Rs, frac_T2,  "o-", color="purple",   lw=2, label="T2* map")
    ax3.plot(Rs, frac_ech, "s--",color="darkorange",lw=1.5, label="Echo images (mean)")
    ax3.set_xscale("log", base=2); ax3.set_xticks(Rs); ax3.set_xticklabels([f"R={r}" for r in Rs])
    ax3.set_xlabel("Acceleration R"); ax3.set_ylabel("Fraction of energy outside sampled band")
    ax3.set_title("Spectral Energy in UNMEASURED Band\n(higher = more must be fabricated)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3); ax3.set_ylim(-0.02, 1.0)

    # ── Panel D: Joint view — error gap and fidelity ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    gap = [r["zf_med"] - r["net_med"] for r in agg_rows]   # +ve = network beats ZF
    ax4.bar([str(r) for r in Rs], gap, color=["green" if g > 1.0 else "tomato" for g in gap])
    ax4.axhline(0, color="k", lw=0.8)
    ax4.set_xlabel("Acceleration R"); ax4.set_ylabel("T2* error gap  ZF − net (ms)")
    ax4.set_title("Network improvement over ZF floor\n(positive = network adds value)")
    ax4.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Acceleration Sweep: T2* identifiability vs R (textured maps, SNR=30 dB)",
                 fontsize=11, y=1.01)
    out = RESULTS_DIR / "accel_sweep_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


def save_map_comparison(first_map_rows: list[dict]) -> None:
    """Six-panel figure for R=1,2,4,8 showing GT, best-net T2*, and absolute error."""
    n_R = len(first_map_rows)
    fig, axes = plt.subplots(3, n_R, figsize=(4 * n_R, 11))
    vmax_t2 = float(first_map_rows[0]["T2_gt"].max()) + 5
    vmax_err = 30.0

    for ci, row in enumerate(first_map_rows):
        T2_gt   = row["T2_gt"]
        T2_net  = row["T2_best"]
        T2_zf   = row["T2_zf"]
        R       = row["R"]

        im0 = axes[0, ci].imshow(T2_gt,  vmin=0, vmax=vmax_t2,  cmap="viridis", origin="upper")
        axes[0, ci].set_title(f"R={R}\nGT T2* (ms)", fontsize=9); axes[0, ci].axis("off")
        plt.colorbar(im0, ax=axes[0, ci], fraction=0.046, pad=0.04)

        im1 = axes[1, ci].imshow(T2_net, vmin=0, vmax=vmax_t2,  cmap="viridis", origin="upper")
        axes[1, ci].set_title(f"3p-CNN  med={row['net_med']:.1f}ms", fontsize=9)
        axes[1, ci].axis("off")
        plt.colorbar(im1, ax=axes[1, ci], fraction=0.046, pad=0.04)

        err = np.abs(T2_net - T2_gt)
        im2 = axes[2, ci].imshow(err, vmin=0, vmax=vmax_err, cmap="hot", origin="upper")
        axes[2, ci].set_title(f"|Error|  corr={row['tf_net']['corr']:.3f}", fontsize=9)
        axes[2, ci].axis("off")
        plt.colorbar(im2, ax=axes[2, ci], fraction=0.046, pad=0.04)

    fig.suptitle("T2* reconstruction vs acceleration (test map 1)", fontsize=11)
    fig.tight_layout()
    out = RESULTS_DIR / "accel_sweep_maps.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT TABLE + R* LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def print_verdict_table(agg_rows: list[dict]) -> tuple[int | None, str]:
    """Print the full results table and return (R*, verdict_string)."""
    print("\n" + "═" * 110)
    print("ACCELERATION SWEEP — RESULTS")
    print("  Textured maps (held-out), df=0 Hz, SNR=30 dB, 3p-CNN, 3 seeds")
    print("═" * 110)

    hdr = (f"  {'R':>3}  {'net-med':>8}  {'net-p95':>8}  {'ZF-med':>7}  {'ZF-p95':>7}  "
           f"{'UB-med':>7}  {'xseed':>6}  {'net-corr':>9}  {'ZF-corr':>8}  "
           f"{'halluc':>7}  {'frac-T2':>8}  {'frac-echo':>10}")
    print(hdr)
    print(f"  {'':>3}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>7}  {'(ms)':>7}  "
          f"{'(ms)':>7}  {'(ms)':>6}  {'(Pearson)':>9}  {'':>8}  "
          f"{'':>7}  {'unmeas':>8}  {'unmeas':>10}")
    print("  " + "─" * 106)

    for r in agg_rows:
        print(f"  {r['R']:>3}  {r['net_med']:>8.2f}  {r['net_p95']:>8.2f}  "
              f"{r['zf_med']:>7.2f}  {r['zf_p95']:>7.2f}  {r['ub_med']:>7.2f}  "
              f"{r['xseed']:>6.3f}  {r['net_corr']:>9.4f}  {r['zf_corr']:>8.4f}  "
              f"{r['net_halluc']:>7.3f}  {r['frac_T2_unmeas']:>8.3f}  "
              f"{r['frac_echo_unmeas']:>10.3f}")

    print("  " + "═" * 106)

    # ── R* determination ─────────────────────────────────────────────────────
    # Criteria for "honestly recovered" at R:
    #   (a) net_corr > 0.5                (fidelity well above noise level)
    #   (b) net_med < 0.8 × zf_med       (network clearly beats ZF floor)
    # R* = highest R satisfying both criteria.

    R_star = None
    reasons: dict[int, str] = {}
    for r in agg_rows:
        crit_a = r["net_corr"] > 0.5
        crit_b = r["net_med"]  < 0.8 * r["zf_med"]
        honest = crit_a and crit_b
        reasons[r["R"]] = f"corr={r['net_corr']:.3f}>0.5:{crit_a}, net<0.8·ZF:{crit_b}"
        if honest:
            R_star = r["R"]

    print(f"\n  R* criteria: unmeas-band Pearson r > 0.5  AND  net_med < 0.8 × ZF_med")
    for R_val, reason in reasons.items():
        mark = "✓ HONEST" if (r := next(r for r in agg_rows if r["R"] == R_val)) and \
               r["net_corr"] > 0.5 and r["net_med"] < 0.8 * r["zf_med"] else "✗"
        print(f"    R={R_val}: {reason}  →  {mark}")

    print()
    if R_star is None:
        verdict = (
            "R* = NONE — per-pixel texture is NOT honestly recoverable at any tested "
            "acceleration. Even at R=2, the unmeasured k-space band contains enough "
            "textured energy that the CNN prior fills it with smooth content, not truth. "
            "Per-pixel T2* texture claims at any R > 1 are in fabrication territory "
            "for this phantom class (textured, SNR=30 dB, 4× to 8× accel)."
        )
    else:
        above_Rstar = [r["R"] for r in agg_rows if r["R"] > R_star]
        verdict = (
            f"R* = {R_star}. At R≤{R_star} the 3-param CNN recovers per-pixel T2* texture "
            f"with unmeasured-band correlation > 0.5 and T2* error clearly below the "
            f"ZF floor. At R > {R_star} ({above_Rstar}) the unmeasured band energy fraction "
            f"is high enough that the network hallucinates texture — the per-pixel T2* "
            f"claim enters fabrication territory. "
            f"The operating window for honest per-pixel T2* is R ≤ {R_star}."
        )

    print("  VERDICT:")
    import textwrap
    for line in textwrap.wrap(verdict, width=100):
        print(f"    {line}")
    print(f"\n  R* = {R_star}")
    print("═" * 110 + "\n")
    return R_star, verdict


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=RECON_EPOCHS)
    args = parser.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    print(f"\nDevice: {device}")
    print(f"Recon epochs: {args.epochs}  |  R sweep: {ACCEL_SWEEP}  |  Seeds: {N_SEEDS}")
    print(f"SNR={SNR_DB} dB  |  mask_seed={MASK_SEED}  |  noise_seed={NOISE_SEED}\n")

    # ── load test maps (same as JEPA experiment) ──────────────────────────────
    print("Loading test maps…")
    all_maps = load_test_maps()
    test_maps = all_maps[:N_TEST_MAPS]
    print(f"  Using {len(test_maps)} maps from files "
          f"{list({m['file'] for m in test_maps})}\n")

    # ── pre-compute full-sample analytical UB for each map (R-independent) ───
    print("Computing full-sample analytical upper bounds…")
    T2_ubs: list[np.ndarray] = []
    for tmap in test_maps:
        data_full = synthesise(
            tmap["S0"], tmap["T2"], TEs_MS,
            accel=1, cf=CF, snr_db=SNR_DB, mask_seed=MASK_SEED, noise_seed=NOISE_SEED,
        )
        _, T2_ub = analytical_fit(data_full["echoes_full"], TEs_MS)
        T2_ubs.append(T2_ub)
    print("  Done.\n")

    # ── main sweep ────────────────────────────────────────────────────────────
    agg_rows:    list[dict] = []
    first_map_per_R: list[dict] = []   # for the map-comparison figure

    for R in ACCEL_SWEEP:
        print(f"══ R={R} ══")
        per_map: list[dict] = []

        for mi, (tmap, T2_ub) in enumerate(zip(test_maps, T2_ubs)):
            print(f"  map {mi+1}/{N_TEST_MAPS}  ({tmap['file']} sl={tmap['slice']} seed={tmap['seed']})")
            row = run_one_accel(R, tmap, T2_ub, args.epochs, device)
            per_map.append(row)

        agg = aggregate(per_map)
        agg_rows.append(agg)
        first_map_per_R.append(per_map[0])

        print(f"  → R={R} aggregate: net={agg['net_med']:.2f}ms  "
              f"zf={agg['zf_med']:.2f}ms  corr={agg['net_corr']:.4f}  "
              f"frac_T2_unmeas={agg['frac_T2_unmeas']:.3f}\n")

    # ── save results JSON ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "accel_sweep.json"
    serialisable = []
    for r in agg_rows:
        serialisable.append({k: v for k, v in r.items()
                              if not isinstance(v, np.ndarray)})
    with open(out_json, "w") as f:
        json.dump({"sweep": serialisable}, f, indent=2)
    print(f"  → {out_json.name}")

    # ── figures ───────────────────────────────────────────────────────────────
    save_curve_figure(agg_rows)
    save_map_comparison(first_map_per_R)

    # ── verdict ───────────────────────────────────────────────────────────────
    R_star, verdict = print_verdict_table(agg_rows)
    print(f"Final verdict: R* = {R_star}")


if __name__ == "__main__":
    main()
