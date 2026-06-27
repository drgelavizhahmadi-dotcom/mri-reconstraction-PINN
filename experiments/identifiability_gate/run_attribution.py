#!/usr/bin/env python3
"""
Attribution experiment: physics vs architecture vs information limit.
=====================================================================
Three arms run on SAME maps/seeds across R ∈ {1,2,4,8},
phantom (piecewise-constant) and textured CORPD maps, df=0.

ARM 1 — PHYSICS-ONLY (no network, no labels)
  (a) ARM1_FULL: analytical per-pixel log-linear fit on FULLY-SAMPLED echoes.
      Upper bound: data is complete, only noise limits accuracy.
  (b) ARM1_ZF:   same fit on ZERO-FILLED undersampled magnitude echoes.
      Physics-only floor for undersampled data.

ARM 2 — PHYSICS-GROUNDED (the method)
  BottleneckNet3MC + complex k-space data-consistency loss.
  Label-free (self-supervised); uses the forward operator.

ARM 3 — NO-PHYSICS BLACK-BOX (steelmanned control)
  IDENTICAL architecture (BottleneckNet3MC, same width/depth/dropout).
  Forward operator REMOVED; trained by SUPERVISED regression to GT maps.
  Given GT labels (an advantage vs ARM 2) so it is not strawmanned.

METRICS per arm / R / map-type:
  A. Recovery:    T2* error (median, p95), texture-fidelity corr (HF band)
  B. Uncertainty: Spearman rho(uncertainty, |error|), boundary/interior std ratio
                  — MC-dropout active on BOTH arms 2 and 3.

Usage:
    python experiments/identifiability_gate/run_attribution.py
"""

from __future__ import annotations

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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, sobel, zoom
from scipy.stats import pearsonr, spearmanr

_GATE_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _GATE_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_GATE_DIR))

from run_gate import (                            # noqa: E402
    synthesise, analytical_fit,
    H, W, TEs_MS, N_ECHOES, CF, HIDDEN, T2_MIN, T2_MAX, RESULTS_DIR,
)
from run_offres_fix import (                      # noqa: E402
    kspace_loss_3param, DF_BOUND, _make_complex_zf_input,
)
from run_calib_uncertainty import (               # noqa: E402
    BottleneckNet3MC, train_mc_model, mc_predict,
    MASK_SEED, NOISE_SEED, SNR_DB, TRAIN_EPOCHS, DROPOUT_P, N_MC,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────── constants ───────────────────────────────────

ACCEL_SWEEP      = [1, 2, 4, 8]
SEED             = 42
N_MAPS           = 2            # CORPD files used (keeps runtime short)
TEXTURE_AMP      = 0.30
TEXTURE_SIGMA    = 3.0
N_PRETRAIN_FILES = 8
DATA_DIR         = Path("data/singlecoil_val")
HF_SIGMA         = 3.0          # Gaussian high-pass σ for texture-fidelity


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — DATA LOADING
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


def _make_slice_maps(
    rss_slice: np.ndarray, texture_seed: int, texture_amp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S0, T2_gt, fg); tissue-class T2* ± Gaussian texture."""
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

    if texture_amp > 0:
        rng      = np.random.default_rng(texture_seed)
        noise    = rng.standard_normal((H, W)).astype(np.float32)
        noise_sm = gaussian_filter(noise, sigma=TEXTURE_SIGMA)
        noise_sm /= noise_sm.std() + 1e-8
        T2 = np.clip(T2_base * (1.0 + texture_amp * noise_sm), T2_MIN, T2_MAX)
    else:
        T2 = T2_base.copy()

    T2[~fg] = T2_MIN
    return S0.astype(np.float32), T2.astype(np.float32), fg


def load_maps() -> dict[str, list[dict]]:
    """
    Load N_MAPS CORPD test files; from each file take one mid-slice.
    Returns phantom (texture_amp=0) and textured (texture_amp=0.3) variants.
    """
    all_files  = _get_corpd_files()
    test_files = all_files[N_PRETRAIN_FILES: N_PRETRAIN_FILES + N_MAPS]
    phantom_maps: list[dict] = []
    textured_maps: list[dict] = []

    for fn in test_files:
        with h5py.File(DATA_DIR / fn, "r") as f:
            rss = f["reconstruction_rss"][()]
        n_sl = rss.shape[0]
        si   = int(n_sl * 0.5)

        for mtype, amp in [("phantom", 0.0), ("textured", TEXTURE_AMP)]:
            S0, T2, fg = _make_slice_maps(rss[si], texture_seed=2, texture_amp=amp)
            rec = dict(S0=S0, T2=T2, fg=fg,
                       Df=np.zeros((H, W), dtype=np.float32),
                       file=fn, slice=int(si), mtype=mtype)
            (phantom_maps if mtype == "phantom" else textured_maps).append(rec)

    return dict(phantom=phantom_maps, textured=textured_maps)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — ARM 3 SUPERVISED TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def _supervised_loss(
    s0_pred: torch.Tensor, t2_pred: torch.Tensor, df_pred: torch.Tensor,
    s0_gt: torch.Tensor,   t2_gt: torch.Tensor,   df_gt: torch.Tensor,
) -> torch.Tensor:
    """MSE on GT maps; each term normalised to [0,1] scale to equalise magnitudes."""
    ls0 = F.mse_loss(s0_pred, s0_gt)
    lt2 = F.mse_loss(t2_pred / T2_MAX, t2_gt / T2_MAX)
    ldf = F.mse_loss(df_pred / DF_BOUND, df_gt / DF_BOUND)
    return ls0 + lt2 + ldf


def train_arm3_supervised(
    model: BottleneckNet3MC,
    zf_input: torch.Tensor,          # [1, 2*N_E, H, W]
    s0_gt: np.ndarray,
    t2_gt: np.ndarray,
    df_gt: np.ndarray,
    n_epochs: int,
    device: torch.device,
    label: str = "",
) -> BottleneckNet3MC:
    """
    ARM 3: identical architecture to ARM 2 but supervised on GT maps.
    No forward operator; loss is purely MAP-space regression.
    Steelmanned: receives GT labels that ARM 2 never sees.
    """
    model    = model.to(device)
    zf_input = zf_input.to(device)
    s0_tc    = torch.from_numpy(s0_gt[None, None]).to(device)   # [1,1,H,W]
    t2_tc    = torch.from_numpy(t2_gt[None, None]).to(device)
    df_tc    = torch.from_numpy(df_gt[None, None]).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    log_ev = max(1, n_epochs // 4)

    for ep in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        s0_p, t2_p, df_p = model(zf_input)
        loss = _supervised_loss(s0_p, t2_p, df_p, s0_tc, t2_tc, df_tc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log_ev == 0:
            print(f"    [{label}] ep {ep:4d}/{n_epochs}  loss={loss.item():.5f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def recovery_metrics(T2_pred: np.ndarray, T2_gt: np.ndarray, fg: np.ndarray) -> dict:
    err = np.abs(T2_pred[fg] - T2_gt[fg])
    return dict(t2_med=float(np.median(err)), t2_p95=float(np.percentile(err, 95)))


def texture_fidelity(
    T2_pred: np.ndarray, T2_gt: np.ndarray, fg: np.ndarray, sigma: float = HF_SIGMA
) -> float:
    """Pearson r of Gaussian-high-pass residuals on fg pixels."""
    hf_pred = T2_pred - gaussian_filter(T2_pred.astype(np.float64), sigma)
    hf_gt   = T2_gt   - gaussian_filter(T2_gt.astype(np.float64),   sigma)
    if fg.sum() < 20:
        return float("nan")
    r, _ = pearsonr(hf_pred[fg], hf_gt[fg])
    return float(r) if np.isfinite(r) else 0.0


def uncertainty_metrics(
    T2_std: np.ndarray, T2_err: np.ndarray, T2_gt: np.ndarray, fg: np.ndarray
) -> dict:
    """Spearman rho(std, |err|) and boundary/interior mean-std ratio."""
    rho, _ = spearmanr(T2_std[fg], T2_err[fg])

    gx   = sobel(T2_gt.astype(np.float64), axis=0)
    gy   = sobel(T2_gt.astype(np.float64), axis=1)
    grad = np.sqrt(gx**2 + gy**2)
    fg_g = grad[fg]; fg_s = T2_std[fg]
    p50  = np.percentile(fg_g, 50)
    p80  = np.percentile(fg_g, 80)
    int_std = fg_s[fg_g <= p50].mean()
    bnd_std = fg_s[fg_g >= p80].mean()
    bnd_int = float(bnd_std / (int_std + 1e-8))

    return dict(rho=float(rho) if np.isfinite(rho) else 0.0, bnd_int=bnd_int)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — RUN ONE CONDITION (one map × one R)
# ═══════════════════════════════════════════════════════════════════════════════

def run_condition(tmap: dict, R: int, device: torch.device, map_idx: int) -> dict:
    S0_gt = tmap["S0"]; T2_gt = tmap["T2"]
    fg    = tmap["fg"]; Df_gt = tmap["Df"]
    pfx   = f"R={R} m{map_idx} ({tmap['mtype']})"

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # ── synthesise ────────────────────────────────────────────────────────────
    data    = synthesise(S0_gt, T2_gt, TEs_MS, accel=max(R, 1),
                         cf=CF, snr_db=SNR_DB, mask_seed=MASK_SEED, noise_seed=NOISE_SEED)

    zf_arr  = _make_complex_zf_input(data["kspace_under"])
    zf_in   = torch.from_numpy(zf_arr).unsqueeze(0)          # [1, 2*N_E, H, W]
    k_under = torch.from_numpy(data["kspace_under"]).to(torch.complex64)
    mask_tc = torch.from_numpy(data["mask_2d"])

    # ── ARM 1 — analytical fit ────────────────────────────────────────────────
    _, T2_ana_full = analytical_fit(data["echoes_full"], TEs_MS)
    _, T2_ana_zf   = analytical_fit(data["zf_mag"],     TEs_MS)

    # ── ARM 2 — physics-grounded (k-space DC loss, self-supervised) ──────────
    arm2 = BottleneckNet3MC(n_echoes=N_ECHOES, hidden=HIDDEN,
                            T2_min=T2_MIN, T2_max=T2_MAX,
                            df_bound=DF_BOUND, p=DROPOUT_P)
    arm2, _ = train_mc_model(arm2, zf_in, k_under, mask_tc,
                             TRAIN_EPOCHS, device, label=f"ARM2 {pfx}")
    T2_arm2_mean, T2_arm2_std = mc_predict(arm2, zf_in, device, n_samples=N_MC)

    # ── ARM 3 — no-physics black-box, supervised on GT maps (steelmanned) ────
    arm3 = BottleneckNet3MC(n_echoes=N_ECHOES, hidden=HIDDEN,
                            T2_min=T2_MIN, T2_max=T2_MAX,
                            df_bound=DF_BOUND, p=DROPOUT_P)
    arm3 = train_arm3_supervised(
        arm3, zf_in, S0_gt, T2_gt, Df_gt,
        n_epochs=TRAIN_EPOCHS, device=device, label=f"ARM3 {pfx}",
    )
    T2_arm3_mean, T2_arm3_std = mc_predict(arm3, zf_in, device, n_samples=N_MC)

    # ── metrics ───────────────────────────────────────────────────────────────
    rec = {
        "arm1_full": recovery_metrics(T2_ana_full,  T2_gt, fg),
        "arm1_zf":   recovery_metrics(T2_ana_zf,    T2_gt, fg),
        "arm2":      recovery_metrics(T2_arm2_mean, T2_gt, fg),
        "arm3":      recovery_metrics(T2_arm3_mean, T2_gt, fg),
    }
    tex = {
        "arm1_full": texture_fidelity(T2_ana_full,  T2_gt, fg),
        "arm1_zf":   texture_fidelity(T2_ana_zf,    T2_gt, fg),
        "arm2":      texture_fidelity(T2_arm2_mean, T2_gt, fg),
        "arm3":      texture_fidelity(T2_arm3_mean, T2_gt, fg),
    }
    err2 = np.abs(T2_arm2_mean - T2_gt)
    err3 = np.abs(T2_arm3_mean - T2_gt)
    unc = {
        "arm2": uncertainty_metrics(T2_arm2_std, err2, T2_gt, fg),
        "arm3": uncertainty_metrics(T2_arm3_std, err3, T2_gt, fg),
    }

    print(f"  [{pfx}] T2* med(ms):  "
          f"A1f={rec['arm1_full']['t2_med']:.1f}  "
          f"A1z={rec['arm1_zf']['t2_med']:.1f}  "
          f"A2={rec['arm2']['t2_med']:.1f}  "
          f"A3={rec['arm3']['t2_med']:.1f}")
    print(f"  [{pfx}] tex-corr:     "
          f"A1f={tex['arm1_full']:.3f}  "
          f"A1z={tex['arm1_zf']:.3f}  "
          f"A2={tex['arm2']:.3f}  "
          f"A3={tex['arm3']:.3f}")
    print(f"  [{pfx}] unc rho:      "
          f"A2={unc['arm2']['rho']:.3f} bnd/int={unc['arm2']['bnd_int']:.2f}  "
          f"A3={unc['arm3']['rho']:.3f} bnd/int={unc['arm3']['bnd_int']:.2f}")

    return dict(R=R, mtype=tmap["mtype"], rec=rec, tex=tex, unc=unc,
                _arrays=dict(T2_gt=T2_gt, T2_ana_full=T2_ana_full,
                             T2_ana_zf=T2_ana_zf, T2_arm2=T2_arm2_mean,
                             T2_arm3=T2_arm3_mean, std_arm2=T2_arm2_std,
                             std_arm3=T2_arm3_std, fg=fg))


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — AGGREGATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _avg(results: list[dict], R: int, mtype: str, arm: str, metric: str) -> float:
    vals = [r["rec"][arm][metric] for r in results if r["R"] == R and r["mtype"] == mtype]
    return float(np.mean(vals)) if vals else float("nan")


def _avg_tex(results: list[dict], R: int, mtype: str, arm: str) -> float:
    vals = [r["tex"][arm] for r in results if r["R"] == R and r["mtype"] == mtype]
    return float(np.mean(vals)) if vals else float("nan")


def _avg_unc(results: list[dict], R: int, mtype: str, arm: str, metric: str) -> float:
    vals = [r["unc"][arm][metric] for r in results if r["R"] == R and r["mtype"] == mtype]
    return float(np.mean(vals)) if vals else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def save_recovery_figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Attribution: T2* recovery and texture-fidelity × R\n"
        "A1-Full=analytical(full-data), A1-ZF=analytical(ZF), "
        "A2=physics-grounded(self-supervised), A3=no-physics(supervised GT)",
        fontsize=9,
    )
    Rs = ACCEL_SWEEP
    arm_styles = {
        "arm1_full": ("A1-Full", "k-",  2.5),
        "arm1_zf":   ("A1-ZF",  "k--", 1.5),
        "arm2":      ("A2",     "b-",  2.0),
        "arm3":      ("A3",     "r--", 2.0),
    }
    unc_styles = {
        "arm2": ("A2 ρ", "b:", 1.4),
        "arm3": ("A3 ρ", "r:", 1.4),
    }
    for col, mtype in enumerate(["phantom", "textured"]):
        ax0 = axes[0, col]
        for arm, (lbl, ls, lw) in arm_styles.items():
            meds = [_avg(results, R, mtype, arm, "t2_med") for R in Rs]
            ax0.plot(Rs, meds, ls, lw=lw, label=lbl, marker="o", ms=5)
        ax0.set_xlabel("R"); ax0.set_ylabel("T2* median error (ms)")
        ax0.set_title(f"{mtype.capitalize()} — T2* median error")
        ax0.set_xticks(Rs); ax0.legend(fontsize=7); ax0.grid(True, alpha=0.3)
        try:
            ax0.set_yscale("log")
        except Exception:
            pass

        ax1 = axes[1, col]
        for arm, (lbl, ls, lw) in arm_styles.items():
            texs = [_avg_tex(results, R, mtype, arm) for R in Rs]
            ax1.plot(Rs, texs, ls, lw=lw, label=lbl, marker="o", ms=5)
        for arm, (lbl, ls, lw) in unc_styles.items():
            rhos = [_avg_unc(results, R, mtype, arm, "rho") for R in Rs]
            ax1.plot(Rs, rhos, ls, lw=lw, alpha=0.75, label=lbl, marker="s", ms=4)
        ax1.axhline(0, color="k", ls="--", lw=0.8)
        ax1.set_xlabel("R"); ax1.set_ylabel("correlation")
        ax1.set_title(f"{mtype.capitalize()} — texture fidelity (solid) + Spearman ρ (dotted)")
        ax1.set_xticks(Rs); ax1.legend(fontsize=6.5); ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "attribution_recovery.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


def save_spatial_figure(res: dict) -> None:
    arr   = res["_arrays"]
    mtype = res["mtype"]; R = res["R"]
    vmax  = float(arr["T2_gt"].max()) + 5.0

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(
        f"Attribution spatial — {mtype}, R={R}\n"
        "Top: T2* maps  Bottom: |error| (rows 1-4) and A2std/A3std ratio (col 5)",
        fontsize=9,
    )
    top_titles = ["GT T2*", "A1-Full\n(analytical)", "A1-ZF\n(analytical ZF)",
                  "A2 physics-grounded\n(self-supervised)", "A3 no-physics\n(supervised GT)"]
    top_imgs   = [arr["T2_gt"], arr["T2_ana_full"], arr["T2_ana_zf"],
                  arr["T2_arm2"], arr["T2_arm3"]]
    for ax, title, img in zip(axes[0], top_titles, top_imgs):
        im = ax.imshow(img, vmin=0, vmax=vmax, cmap="viridis", origin="upper")
        ax.set_title(title, fontsize=8); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    bot_titles = ["|err| A1-Full", "|err| A1-ZF", "|err| A2", "|err| A3",
                  "A2-std / A3-std\n(>1 = A2 more uncertain)"]
    bot_imgs   = [
        np.abs(arr["T2_ana_full"] - arr["T2_gt"]),
        np.abs(arr["T2_ana_zf"]   - arr["T2_gt"]),
        np.abs(arr["T2_arm2"]     - arr["T2_gt"]),
        np.abs(arr["T2_arm3"]     - arr["T2_gt"]),
        arr["std_arm2"] / (arr["std_arm3"] + 1e-4),
    ]
    bot_vmaxs = [30, 30, 30, 30, 3.0]
    bot_cmaps = ["hot", "hot", "hot", "hot", "RdBu_r"]
    for ax, title, img, vm, cm in zip(axes[1], bot_titles, bot_imgs, bot_vmaxs, bot_cmaps):
        im = ax.imshow(img, vmin=0, vmax=vm, cmap=cm, origin="upper")
        ax.set_title(title, fontsize=8); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = RESULTS_DIR / f"attribution_{mtype}_R{R}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7 — VERDICT TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def print_verdict(results: list[dict]) -> str:
    import textwrap

    # ── closure helpers ───────────────────────────────────────────────────────
    def agg(mtype: str, Rs: list[int], arm: str, metric: str) -> float:
        vals = [_avg(results, R, mtype, arm, metric) for R in Rs]
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else float("nan")

    def agg_tex(mtype: str, Rs: list[int], arm: str) -> float:
        vals = [_avg_tex(results, R, mtype, arm) for R in Rs]
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else float("nan")

    def agg_unc(mtype: str, Rs: list[int], arm: str, metric: str) -> float:
        vals = [_avg_unc(results, R, mtype, arm, metric) for R in Rs]
        clean = [v for v in vals if np.isfinite(v)]
        return float(np.mean(clean)) if clean else float("nan")

    R_low  = [R for R in ACCEL_SWEEP if R <= 2]
    R_high = [R for R in ACCEL_SWEEP if R > 2]
    Rs_all = ACCEL_SWEEP

    print("\n" + "═" * 120)
    print("ATTRIBUTION VERDICT TABLES")
    print("═" * 120)

    # ── TABLE A: Recovery ─────────────────────────────────────────────────────
    for mtype in ["phantom", "textured"]:
        print(f"\n  TABLE A — T2* median error (ms)  [{mtype.upper()}]")
        print(f"  {'R':>3}  {'A1-Full':>9}  {'A1-ZF':>9}  {'A2-phys':>9}  {'A3-bb':>9}"
              f"  {'A2-p95':>9}  {'A3-p95':>9}")
        print(f"  {'─'*3}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
        for R in ACCEL_SWEEP:
            print(f"  {R:>3}  "
                  f"{_avg(results,R,mtype,'arm1_full','t2_med'):>9.2f}  "
                  f"{_avg(results,R,mtype,'arm1_zf',  't2_med'):>9.2f}  "
                  f"{_avg(results,R,mtype,'arm2',     't2_med'):>9.2f}  "
                  f"{_avg(results,R,mtype,'arm3',     't2_med'):>9.2f}  "
                  f"{_avg(results,R,mtype,'arm2',     't2_p95'):>9.2f}  "
                  f"{_avg(results,R,mtype,'arm3',     't2_p95'):>9.2f}")

    # ── TABLE B: Texture fidelity ─────────────────────────────────────────────
    for mtype in ["phantom", "textured"]:
        print(f"\n  TABLE B — Texture fidelity (Pearson r, HF band)  [{mtype.upper()}]")
        print(f"  {'R':>3}  {'A1-Full':>9}  {'A1-ZF':>9}  {'A2-phys':>9}  {'A3-bb':>9}")
        print(f"  {'─'*3}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
        for R in ACCEL_SWEEP:
            print(f"  {R:>3}  "
                  f"{_avg_tex(results,R,mtype,'arm1_full'):>9.4f}  "
                  f"{_avg_tex(results,R,mtype,'arm1_zf'):>9.4f}  "
                  f"{_avg_tex(results,R,mtype,'arm2'):>9.4f}  "
                  f"{_avg_tex(results,R,mtype,'arm3'):>9.4f}")

    # ── TABLE C: Uncertainty quality ─────────────────────────────────────────
    for mtype in ["phantom", "textured"]:
        print(f"\n  TABLE C — Uncertainty quality  [{mtype.upper()}]")
        print(f"  {'R':>3}  {'A2 ρ':>8}  {'A2 bnd/int':>11}  {'A3 ρ':>8}  {'A3 bnd/int':>11}"
              f"  {'Δρ(A2-A3)':>11}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*11}  {'─'*8}  {'─'*11}  {'─'*11}")
        for R in ACCEL_SWEEP:
            a2r = _avg_unc(results, R, mtype, "arm2", "rho")
            a3r = _avg_unc(results, R, mtype, "arm3", "rho")
            a2b = _avg_unc(results, R, mtype, "arm2", "bnd_int")
            a3b = _avg_unc(results, R, mtype, "arm3", "bnd_int")
            print(f"  {R:>3}  {a2r:>8.4f}  {a2b:>11.3f}  {a3r:>8.4f}  {a3b:>11.3f}"
                  f"  {a2r-a3r:>+11.4f}")

    # ── KEY QUESTIONS ─────────────────────────────────────────────────────────
    print("\n" + "─" * 120)
    print("  KEY QUESTIONS")
    print("─" * 120)

    print("\n  Q1: At R=1, does ARM1_FULL (noise-only limit) beat the trained networks?")
    for mtype in ["phantom", "textured"]:
        a1f = _avg(results, 1, mtype, "arm1_full", "t2_med")
        a2  = _avg(results, 1, mtype, "arm2",      "t2_med")
        a3  = _avg(results, 1, mtype, "arm3",      "t2_med")
        print(f"    {mtype}: A1-Full={a1f:.2f}  A2={a2:.2f}  A3={a3:.2f} ms  → "
              + ("A1-FULL BEST (all information available; network overhead beats pure math)" if a1f < a2 and a1f < a3
                 else "NETWORK MATCHES/BEATS A1-FULL at R=1 (deep prior useful even at full sampling)"))

    print("\n  Q2: At R≤2, do ARM2 and ARM3 match (information-constrained regime)?")
    for mtype in ["phantom", "textured"]:
        a2 = agg(mtype, R_low, "arm2", "t2_med")
        a3 = agg(mtype, R_low, "arm3", "t2_med")
        delta = abs(a2 - a3)
        print(f"    {mtype} (R≤2): A2={a2:.2f}  A3={a3:.2f}  Δ={delta:.2f} ms  → "
              + ("MATCH (<3 ms) — recovery limited by measurement, not physics grounding"
                 if delta < 3.0
                 else "DIVERGE (≥3 ms) — physics grounding adds recoverable information at R≤2"))

    print("\n  Q3: At R>2, does ARM3 (no-physics, supervised) also fabricate texture?")
    for mtype in ["textured"]:
        a2 = agg_tex(mtype, R_high, "arm2")
        a3 = agg_tex(mtype, R_high, "arm3")
        print(f"    {mtype} (R>2): A2 tex-r={a2:.4f}  A3 tex-r={a3:.4f}  → "
              + ("BOTH FABRICATE (tex-r < 0.5) — blind spot is GENERAL, not physics-specific"
                 if a2 < 0.5 and a3 < 0.5
                 else "ONE FABRICATES MORE — "
                      + ("physics grounding mitigates fabrication" if a2 > a3
                         else "supervised arm recovers texture better (label advantage dominates)")))

    print("\n  Q4: Does physics grounding improve uncertainty ranking over no-physics arm?")
    for mtype in ["phantom", "textured"]:
        a2r = agg_unc(mtype, Rs_all, "arm2", "rho")
        a3r = agg_unc(mtype, Rs_all, "arm3", "rho")
        delta = a2r - a3r
        print(f"    {mtype}: A2 ρ={a2r:.4f}  A3 ρ={a3r:.4f}  Δρ={delta:+.4f}  → "
              + ("PHYSICS WINS (Δρ > 0.05) — forward operator informs uncertainty structure"
                 if delta > 0.05
                 else "ARCHITECTURE (|Δρ| ≤ 0.05) — MC-dropout, not physics, drives ranking quality"))

    # ── PER-CLAIM ATTRIBUTION VERDICT ─────────────────────────────────────────
    print("\n" + "═" * 120)
    print("  PER-CLAIM ATTRIBUTION VERDICT")
    print("═" * 120)

    # aggregate numbers for verdict text
    a2_lo = float(np.mean([agg(m, R_low, "arm2", "t2_med") for m in ["phantom","textured"]]))
    a3_lo = float(np.mean([agg(m, R_low, "arm3", "t2_med") for m in ["phantom","textured"]]))
    a2_hi_tex = float(np.mean([agg_tex(m, R_high, "arm2") for m in ["phantom","textured"]]))
    a3_hi_tex = float(np.mean([agg_tex(m, R_high, "arm3") for m in ["phantom","textured"]]))
    drho_list = [agg_unc(m, Rs_all, "arm2", "rho") - agg_unc(m, Rs_all, "arm3", "rho")
                 for m in ["phantom", "textured"]]
    drho_mean = float(np.mean(drho_list))

    claim1 = (
        f"CLAIM 1 — IDENTIFIABILITY AT R≤{max(R_low)}: "
        f"ARM2 (physics-grounded, label-free) achieves T2* median {a2_lo:.2f} ms vs "
        f"ARM3 (no-physics, supervised GT) {a3_lo:.2f} ms at R≤{max(R_low)}. "
        f"Δ = {abs(a2_lo-a3_lo):.2f} ms. "
        + ("ATTRIBUTION: INFORMATION LIMIT — at low R the measurement constrains recovery "
           "equally for both arms. Physics grounding provides self-supervision (no GT needed) "
           "but does not add recoverable information beyond what the data contain."
           if abs(a2_lo-a3_lo) < 3.0
           else "ATTRIBUTION: PHYSICS-SPECIFIC — the forward operator adds information even at R≤2; "
                "self-supervision outperforms supervised arm that lacks the physics prior.")
    )

    blind_verdict = (
        "ATTRIBUTION: GENERAL INFORMATION LIMIT — any method (physics-grounded or pure regression) "
        "fabricates T2* texture when undersampling removes more than ~50% of k-space. "
        "The failure is not physics-specific; a reviewer claiming physics grounding avoids fabrication "
        "at high R is wrong."
        if a2_hi_tex < 0.5 and a3_hi_tex < 0.5
        else (
            "ATTRIBUTION: PHYSICS-MITIGATED — physics grounding partially reduces fabrication "
            f"(A2 tex-r={a2_hi_tex:.3f} vs A3 tex-r={a3_hi_tex:.3f})."
            if a2_hi_tex > a3_hi_tex
            else "ATTRIBUTION: LABEL ADVANTAGE — supervised arm recovers more texture at high R "
                 "(GT label signal overrides information limit). Physics grounding shows no advantage."
        )
    )
    claim2 = (
        f"CLAIM 2 — DETECTION BLIND SPOT AT R>{min(R_high)}: "
        f"A2 tex-r={a2_hi_tex:.3f}  A3 tex-r={a3_hi_tex:.3f}. "
        + blind_verdict
    )

    unc_verdict = (
        f"ATTRIBUTION: ARCHITECTURE + MC-DROPOUT — Spearman ρ is statistically "
        f"similar between ARM2 and ARM3 (mean Δρ = {drho_mean:+.4f}, |Δρ| ≤ 0.05). "
        "Ranking-calibrated abstention is a property of the MC-dropout architecture, "
        "independent of physics grounding. The claim can be made for the architecture, "
        "not exclusively for the physics prior."
        if abs(drho_mean) <= 0.05
        else (
            f"ATTRIBUTION: PHYSICS GROUNDING — ARM2 improves Spearman ρ by {drho_mean:+.4f} "
            "over ARM3 (>0.05 margin). The forward operator informs uncertainty structure "
            "beyond what dropout alone provides."
            if drho_mean > 0
            else f"ATTRIBUTION: LABEL ADVANTAGE — supervised ARM3 achieves higher Spearman ρ "
                 f"than physics-grounded ARM2 (Δρ = {drho_mean:+.4f}). GT labels during training "
                 "produce better-ranked uncertainty."
        )
    )
    claim3 = (
        f"CLAIM 3 — RANKING-CALIBRATED ABSTENTION: mean Δρ(ARM2−ARM3) = {drho_mean:+.4f}. "
        + unc_verdict
    )

    verdict_str = "\n\n".join([claim1, claim2, claim3])
    for ci, claim in enumerate([claim1, claim2, claim3], 1):
        print(f"\n  CLAIM {ci}:")
        for line in textwrap.wrap(claim, width=115):
            print(f"    {line}")

    print("\n" + "═" * 120)
    return verdict_str


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

class _NpEnc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_):   return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        try:
            return float(obj)
        except Exception:
            return super().default(obj)


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    print(f"\nDevice: {device}")
    print(f"R sweep: {ACCEL_SWEEP}   TRAIN_EPOCHS={TRAIN_EPOCHS}   N_MC={N_MC}")
    print("ARM 2: physics-grounded (k-space DC loss, self-supervised)")
    print("ARM 3: no-physics black-box (supervised MSE to GT maps, steelmanned)\n")

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading maps …")
    maps = load_maps()
    print(f"  phantom: {len(maps['phantom'])}   textured: {len(maps['textured'])}\n")

    results: list[dict] = []
    saved_fig: set = set()

    for mtype in ["phantom", "textured"]:
        for R in ACCEL_SWEEP:
            print(f"\n{'─'*60}\n{mtype.upper()}  R = {R}\n{'─'*60}")
            for mi, tmap in enumerate(maps[mtype]):
                res = run_condition(tmap, R, device, mi)
                results.append(res)
                key = (mtype, R)
                if key not in saved_fig:
                    save_spatial_figure(res)
                    saved_fig.add(key)

    save_recovery_figure(results)
    verdict = print_verdict(results)

    # strip raw arrays before serialising
    results_clean = [{k: v for k, v in r.items() if k != "_arrays"} for r in results]
    with open(RESULTS_DIR / "attribution.json", "w") as f:
        json.dump(dict(results=results_clean, verdict=verdict), f, indent=2, cls=_NpEnc)
    print("\n  → attribution.json")


if __name__ == "__main__":
    main()
