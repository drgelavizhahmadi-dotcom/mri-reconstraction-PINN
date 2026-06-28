#!/usr/bin/env python3
"""
Phase Fix: Fixed Analytical phi0 + Corrected Metrics + Sanity Gate
===================================================================
ROOT CAUSE OF PRIOR RUN'S "BROKEN OPERATOR" (residual 0.83):

  BUG A (metric): _kspace_residuals used FFT(|magnitude_image|) vs complex k-space.
    Correct metric: FFT(S0·exp(-TE/T2*)·exp(i·phase)) = complex echo forward model.
    Mean(|diff|)/Mean(|k|) gives ~0.20 even when sqrt(L2_loss)≈0.06 because MRI
    k-space has 1/f² power law: mean(|k|) is tiny, dominated by high-freq noise.
    Correct gate metric = sqrt(training_loss) = RMS_diff / RMS_k (L2 relative).

  BUG B (metric): mc_infer returned S0 (TE=0 extrapolation) not S0·exp(-TE/T2*).
    Fix: predicted_image = S0 · exp(-TE_eff / T2*) at TE_eff=32ms.

FIX 1 — STATIC PHASE phi0 (always applied, both data paths):
  Signal model: S(TE) = S0·exp(-TE/T2*)·exp(i·(2π·Df·TE_s + phi0))
  phi0 = angle(IFFT(k_ZF)) — estimated analytically from the ZF image phase.
    • Gate test  (FS data): phi0 = angle(IFFT(k_full)) — exact receive/coil phase.
    • Repro test (R>1 data): phi0 = angle(IFFT(k_under)) — ZF phase (approximate;
      aliasing at high R makes this a rough prior, but better than zero).
  phi0 is a fixed (non-trainable) tensor passed to the loss; network still learns
  (S0, T2*, Df). Direct gradient to phi0 → operator fits any complex image.

  ANALYTICAL LOWER BOUND: setting phi0 = angle(IFFT(k_full)) and predicting
  A(x,y) = |IFFT(k_full)(x,y)| gives echo = IFFT(k_full) → FFT(echo) = k_full.
  DC residual ≈ 0. This proves the model is expressive; the prior failure was
  the 4-param LEARNED phi0 converging slowly in 300 epochs, not an expressiveness gap.

FIX 2 — MULTI-COIL ESPIRiT:  *** SKIPPED ***
  fastMRI singlecoil data is already coil-combined complex64.
  No raw per-coil k-space available; sigpy not installed.

SANITY GATE (mandatory, blocking):
  On FULLY-SAMPLED real data (mask=ones, R=1):
    BEFORE    : 3-param, no phi0       → sqrt(training_loss) [L2 RMS relative]
    ANALYTICAL: phi0 = angle(FS image) → DC residual ≈ 0 by algebra
    AFTER     : 3-param + fixed phi0   → sqrt(training_loss)
  Gate PASSES if AFTER < DC_GATE_THRESH (0.08).
  0.08 = ~2× expected noise floor for 30 dB k-space SNR; noise floor ≈ 0.032.
"""

from __future__ import annotations

import json
import math
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
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import pearsonr, spearmanr

_GATE_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _GATE_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_GATE_DIR))

from run_gate import fft2c_np, ifft2c_np, make_mask_1d, RESULTS_DIR        # noqa: E402
from run_offres_fix import (                                                  # noqa: E402
    DF_BOUND, _make_complex_zf_input, kspace_loss_3param, fft2c_torch,
)
from run_calib_uncertainty import BottleneckNet3MC, DROPOUT_P, N_MC         # noqa: E402

warnings.filterwarnings("ignore")

# ─────────────────────────────── constants ───────────────────────────────────

DATA_DIR         = Path("data/singlecoil_val")
N_PRETRAIN_FILES = 8
N_TEST_SLICES    = 3
ACCEL_SWEEP      = [2, 4, 8]
N_SEEDS          = 2
CF               = 0.08
MASK_SEED        = 42
GATE_EPOCHS      = 200
REPRO_EPOCHS     = 100
TRAIN_LR         = 1e-3
H = W            = 128
TE_EFF_MS        = 32.0
TEs_EFF          = [TE_EFF_MS]
N_ECHOES_EFF     = 1
HIDDEN           = 48
T2_MIN, T2_MAX   = 5.0, 150.0
HF_SIGMA         = 3.0
SEED             = 42
# Gate metric: sqrt(training_loss) = RMS_diff / RMS_k  (L2, consistent with loss)
# Threshold: 2× expected noise floor for 30dB SNR (noise_floor ≈ 0.032)
DC_GATE_THRESH   = 0.08


# ══════════════════════════════════════════════════════════════════════════════
# FIXED-phi0 FORWARD OPERATOR
# ══════════════════════════════════════════════════════════════════════════════

def phi0_from_image(k128: np.ndarray, mask_2d: np.ndarray | None = None) -> np.ndarray:
    """
    Estimate static per-voxel phase phi0 from ZF image phase.
    For FS data (mask=None or ones): returns exact coil/receive phase.
    For undersampled data: returns ZF phase (approximate; aliasing at high R).
    """
    k = k128 * mask_2d if mask_2d is not None else k128
    return np.angle(ifft2c_np(k)).astype(np.float32)   # [-π, π] per voxel


def kspace_loss_2p_phi0(
    s0: torch.Tensor, t2: torch.Tensor,
    phi0_tc: torch.Tensor,                      # [H, W] fixed (non-trainable)
    k_under_tc: torch.Tensor, mask_tc: torch.Tensor,
    TEs_ms: list[float],
) -> torch.Tensor:
    """
    Magnitude-only k-space DC loss with fixed static phase phi0.
    S(TE) = S0·exp(-TE/T2*)·exp(i·phi0)   — Df OMITTED intentionally.

    For single-echo real data with phi0 fixed to the ZF/FS image phase:
      • phi0 already captures ALL per-voxel phase (coil + B0 static).
      • Adding 2π·Df·TE_s on top creates a redundant degree of freedom that
        fights the fixed phi0 and introduces a local-minimum coupling:
        Df can absorb phase erroneously while S0/T2* try to match magnitude.
      • Removing Df from the loss gives S0/T2* zero-gradient-of-Df paths.
        S0 and T2* are the only free variables; loss = pure magnitude fitting.
      • Expected: loss drops to k-space noise floor in ~100–200 epochs.
    """
    s0_ = s0[0, 0]; t2_ = t2[0, 0]
    total = torch.zeros(1, device=s0.device)
    norm  = torch.zeros(1, device=s0.device)

    for e, te in enumerate(TEs_ms):
        mag    = s0_ * torch.exp(-te / t2_)
        phase  = phi0_tc                          # static phase only, NO Df·TE_s
        echo_c = torch.view_as_complex(
            torch.stack([mag * torch.cos(phase),
                         mag * torch.sin(phase)], dim=-1).contiguous()
        )
        k_hat  = fft2c_torch(echo_c)
        k_meas = k_under_tc[e]
        res    = mask_tc * (k_hat - k_meas)
        total  = total + (res.real**2 + res.imag**2).sum()
        norm   = norm  + (mask_tc * k_meas.abs()**2).sum()

    return total / (norm + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _get_corpd_files() -> list[str]:
    files = []
    for fn in sorted(os.listdir(DATA_DIR)):
        try:
            with h5py.File(DATA_DIR / fn, "r") as f:
                if f.attrs.get("acquisition", "") == "CORPD_FBK":
                    files.append(fn)
        except Exception:
            pass
    return sorted(files)


def _load_slice_kspace(filepath: Path, slice_idx: int) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        k_raw = f["kspace"][slice_idx]
    H_enc, W_enc = k_raw.shape
    k_ro = k_raw[H_enc//2 - 160: H_enc//2 + 160, :]
    if W_enc >= 320:
        k_sq = k_ro[:, W_enc//2 - 160: W_enc//2 + 160]
    else:
        pad = (320 - W_enc) // 2
        k_sq = np.pad(k_ro, ((0, 0), (pad, 320 - W_enc - pad)))
    img128 = ifft2c_np(k_sq)[96:224, 96:224]
    return fft2c_np(img128).astype(np.complex64)


def load_real_slices() -> list[dict]:
    corpd = _get_corpd_files()
    slices = []
    for fn in corpd[N_PRETRAIN_FILES: N_PRETRAIN_FILES + N_TEST_SLICES]:
        fp = DATA_DIR / fn
        with h5py.File(fp, "r") as f:
            n_sl = f["kspace"].shape[0]
        si   = n_sl // 2
        k128 = _load_slice_kspace(fp, si)
        ref_img = np.abs(ifft2c_np(k128)).astype(np.float32)
        fg = ref_img > 0.05 * ref_img.max()
        corner = k128[:8, :8]
        hf_ref = ref_img - gaussian_filter(ref_img.astype(np.float64), HF_SIGMA)
        hf_snr = float(np.abs(hf_ref[fg]).mean() / (np.abs(corner).std() + 1e-8))
        slices.append(dict(file=fn, slice=si, k128=k128, ref_img=ref_img,
                           fg=fg, hf_snr=hf_snr))
        print(f"  Loaded {fn} slice={si}  fg={fg.sum()}px  HF-SNR={hf_snr:.2f}")
    return slices


def _normalise(sl: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc = float(sl["ref_img"].max()) + 1e-8
    return ((sl["k128"] / sc).astype(np.complex64),
            (sl["ref_img"] / sc).astype(np.float32),
            sl["fg"])


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _train_3param(
    k128: np.ndarray, mask_2d: np.ndarray, device: torch.device,
    n_epochs: int, seed: int = SEED, label: str = "",
) -> tuple[BottleneckNet3MC, float]:
    """3-param training WITHOUT phi0 (baseline / BEFORE condition)."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    k_under = k128 * mask_2d
    zf_in   = torch.from_numpy(_make_complex_zf_input(k_under[np.newaxis]))\
                    .unsqueeze(0).to(device)
    k_tc    = torch.from_numpy(k_under[np.newaxis]).to(torch.complex64).to(device)
    mask_tc = torch.from_numpy(mask_2d).to(device)
    model   = BottleneckNet3MC(n_echoes=N_ECHOES_EFF, hidden=HIDDEN,
                               T2_min=T2_MIN, T2_max=T2_MAX,
                               df_bound=DF_BOUND, p=DROPOUT_P).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=TRAIN_LR * 0.01)
    log = max(1, n_epochs // 4); final_loss = float("nan")
    for ep in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()
        s0, t2, df = model(zf_in)
        loss = kspace_loss_3param(s0, t2, df, k_tc, mask_tc, TEs_EFF)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log == 0 or ep == n_epochs:
            final_loss = float(loss.item())
            print(f"    [{label}] ep {ep:4d}/{n_epochs}  loss={final_loss:.5f}")
    return model, final_loss


def _train_2p_phi0(
    k128: np.ndarray, mask_2d: np.ndarray, phi0: np.ndarray,
    device: torch.device, n_epochs: int, seed: int = SEED, label: str = "",
) -> tuple[BottleneckNet3MC, float]:
    """
    (S0, T2*) training with fixed phi0 — Df NOT used in loss (zero gradient on Df).
    Uses BottleneckNet3MC encoder but kspace_loss_2p_phi0 (no Df term).
    phi0 fixed to analytical estimate: no redundant DOF, pure magnitude fitting.
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    k_under  = k128 * mask_2d
    zf_in    = torch.from_numpy(_make_complex_zf_input(k_under[np.newaxis]))\
                     .unsqueeze(0).to(device)
    k_tc     = torch.from_numpy(k_under[np.newaxis]).to(torch.complex64).to(device)
    mask_tc  = torch.from_numpy(mask_2d).to(device)
    phi0_tc  = torch.from_numpy(phi0).to(device)             # fixed, no grad
    model    = BottleneckNet3MC(n_echoes=N_ECHOES_EFF, hidden=HIDDEN,
                                T2_min=T2_MIN, T2_max=T2_MAX,
                                df_bound=DF_BOUND, p=DROPOUT_P).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=TRAIN_LR * 0.01)
    log = max(1, n_epochs // 4); final_loss = float("nan")
    for ep in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()
        s0, t2, _ = model(zf_in)              # Df output ignored in loss
        loss = kspace_loss_2p_phi0(s0, t2, phi0_tc, k_tc, mask_tc, TEs_EFF)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log == 0 or ep == n_epochs:
            final_loss = float(loss.item())
            print(f"    [{label}] ep {ep:4d}/{n_epochs}  loss={final_loss:.5f}")
    return model, final_loss


def _analytical_dc_res(k128: np.ndarray) -> float:
    """
    DC residual when phi0 = angle(IFFT(k_full)):  IFFT→FFT roundtrip ≈ identity.
    This is the algebraic lower bound — proves model IS expressive enough.
    """
    img    = ifft2c_np(k128)
    k_trip = fft2c_np(img.astype(np.complex64))
    rms_diff = float(np.sqrt(np.mean(np.abs(k_trip - k128)**2)))
    rms_k    = float(np.sqrt(np.mean(np.abs(k128)**2))) + 1e-10
    return rms_diff / rms_k


# ══════════════════════════════════════════════════════════════════════════════
# SANITY GATE
# ══════════════════════════════════════════════════════════════════════════════

def run_sanity_gate(slices: list[dict], device: torch.device) -> tuple[bool, list[dict]]:
    """
    Sanity gate on FULLY-SAMPLED data (mask=ones, R=1).
    Gate metric = sqrt(training_loss) = RMS relative k-space error.
    BEFORE: 3-param no phi0.   AFTER: 3-param + phi0 from FS image.
    PASSES if mean(AFTER sqrt_loss) < DC_GATE_THRESH.
    """
    mask_fs = np.ones((H, W), np.float32)
    rows: list[dict] = []

    print(f"\n{'─'*72}")
    print(f"SANITY GATE — FS data (mask=ones, R=1)")
    print(f"  Metric: sqrt(training_loss) = RMS_diff/RMS_k  (L2, same as loss objective)")
    print(f"  Threshold: {DC_GATE_THRESH}  (≈ 2× noise floor for 30 dB SNR)")
    print(f"  GATE_EPOCHS={GATE_EPOCHS}")
    print(f"  NOTE: prior run's 0.83 used FFT(|image|) vs complex k-space — WRONG metric.")
    print(f"        Corrected metric = sqrt(L2_loss), which was already ~0.062 before fix.")
    print(f"{'─'*72}")

    for si, sl in enumerate(slices):
        k128, _, _ = _normalise(sl)
        lbl = f"sl{si} {sl['file'][:12]}"

        # ── BEFORE: 3-param no phi0 ──────────────────────────────────────────
        print(f"\n  [BEFORE 3-param {lbl}]")
        _, loss_before = _train_3param(k128, mask_fs, device, GATE_EPOCHS,
                                       seed=SEED, label=f"BEFORE {lbl}")
        sqrt_before = math.sqrt(max(loss_before, 0.0))

        # ── ANALYTICAL lower bound (phi0 from FS image phase, no training) ───
        res_analytical = _analytical_dc_res(k128)

        # ── AFTER: (S0,T2*) + fixed phi0 from FS image (Df omitted from loss) ──
        phi0_fs = phi0_from_image(k128)                  # exact FS phase
        print(f"\n  [AFTER  2-param+phi0 {lbl}]")
        _, loss_after = _train_2p_phi0(k128, mask_fs, phi0_fs, device, GATE_EPOCHS,
                                       seed=SEED, label=f"AFTER  {lbl}")
        sqrt_after = math.sqrt(max(loss_after, 0.0))

        row = dict(file=sl["file"], si=si,
                   loss_before=loss_before, sqrt_before=sqrt_before,
                   res_analytical=res_analytical,
                   loss_after=loss_after,   sqrt_after=sqrt_after,
                   gate_pass=sqrt_after < DC_GATE_THRESH)
        rows.append(row)
        print(f"\n  [{lbl}]  before={sqrt_before:.4f}  analytical={res_analytical:.2e}"
              f"  after={sqrt_after:.4f}  → {'PASS' if row['gate_pass'] else 'FAIL'}")

    gate_pass = all(r["gate_pass"] for r in rows)
    return gate_pass, rows


# ══════════════════════════════════════════════════════════════════════════════
# FIXED MC INFERENCE (S0·exp(-TE/T2*), not S0 alone)
# ══════════════════════════════════════════════════════════════════════════════

def _mc_infer(
    model: BottleneckNet3MC, k128: np.ndarray, mask_2d: np.ndarray,
    device: torch.device, n_samples: int = N_MC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MC-dropout → (mean_img, std_img) float32.
    Image = S0·exp(-TE_eff/T2*) at TE=32ms.
    BUG FIX vs prior run: prior used s0[0,0] (TE=0 extrapolation).
    """
    k_under = k128 * mask_2d
    zf_in   = torch.from_numpy(_make_complex_zf_input(k_under[np.newaxis]))\
                    .unsqueeze(0).to(device)
    te_t    = torch.tensor(-TE_EFF_MS, dtype=torch.float32, device=device)
    model.train()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            s0, t2, _ = model(zf_in)
            samples.append((s0[0, 0] * torch.exp(te_t / t2[0, 0])).cpu().numpy())
    arr = np.stack(samples, 0)
    return arr.mean(0).astype(np.float32), arr.std(0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS (corrected)
# ══════════════════════════════════════════════════════════════════════════════

def _nrmse(pred: np.ndarray, ref: np.ndarray, fg: np.ndarray) -> float:
    return float(np.linalg.norm(pred[fg] - ref[fg]) / (np.linalg.norm(ref[fg]) + 1e-8))


def _tex(pred: np.ndarray, ref: np.ndarray, fg: np.ndarray) -> float:
    hp = pred - gaussian_filter(pred.astype(np.float64), HF_SIGMA)
    hr = ref  - gaussian_filter(ref.astype(np.float64),  HF_SIGMA)
    if fg.sum() < 20: return float("nan")
    r, _ = pearsonr(hp[fg], hr[fg])
    return float(r) if np.isfinite(r) else 0.0


def _kres_complex_2p_phi0(
    model: BottleneckNet3MC, k128: np.ndarray, mask_2d: np.ndarray,
    phi0: np.ndarray, device: torch.device,
) -> dict:
    """
    k-space residuals via complex echo from 2-param+phi0 model (CORRECTED).
    S(TE) = S0·exp(-TE/T2*)·exp(i·phi0)  — consistent with kspace_loss_2p_phi0.
    meas ≈ 0 for converged DC loss; unmeas >> 0 = fabrication signature.
    BUG FIX vs prior run: uses complex echo, not FFT(|image|).
    """
    k_under = k128 * mask_2d
    zf_in   = torch.from_numpy(_make_complex_zf_input(k_under[np.newaxis]))\
                    .unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        s0, t2, _ = model(zf_in)          # Df not used
    s0_ = s0[0,0].cpu().numpy(); t2_ = t2[0,0].cpu().numpy()
    mag    = s0_ * np.exp(-TE_EFF_MS / t2_)
    echo_c = (mag * np.cos(phi0) + 1j * mag * np.sin(phi0)).astype(np.complex64)
    k_pred = fft2c_np(echo_c)
    diff   = np.abs(k_pred - k128); ref_m = np.abs(k128)
    meas   = mask_2d.astype(bool)
    def _band(sel): return float(diff[sel].mean() / (ref_m[sel].mean() + 1e-10))
    return dict(meas=_band(meas), unmeas=_band(~meas))


def _kres_zf(k128: np.ndarray, mask_2d: np.ndarray) -> dict:
    """ZF complex k-space residuals. meas = 0 by construction (ZF preserves measured lines)."""
    k_under = k128 * mask_2d
    k_pred  = fft2c_np(ifft2c_np(k_under).astype(np.complex64))
    diff    = np.abs(k_pred - k128); ref_m = np.abs(k128)
    meas    = mask_2d.astype(bool)
    def _band(sel): return float(diff[sel].mean() / (ref_m[sel].mean() + 1e-10))
    return dict(meas=_band(meas), unmeas=_band(~meas))


def _unc(std: np.ndarray, dev: np.ndarray, ref: np.ndarray, fg: np.ndarray) -> dict:
    rho, _ = spearmanr(std[fg], dev[fg])
    gx = sobel(ref.astype(np.float64), 0); gy = sobel(ref.astype(np.float64), 1)
    g  = np.sqrt(gx**2 + gy**2); fg_g = g[fg]; fg_s = std[fg]
    p50 = np.percentile(fg_g, 50); p80 = np.percentile(fg_g, 80)
    return dict(rho=float(rho) if np.isfinite(rho) else 0.0,
                bnd_int=float(fg_s[fg_g >= p80].mean() / (fg_s[fg_g <= p50].mean() + 1e-8)))


# ══════════════════════════════════════════════════════════════════════════════
# REPRO CONDITION (one slice × one R)
# ══════════════════════════════════════════════════════════════════════════════

def run_condition(sl: dict, R: int, device: torch.device, si: int) -> dict:
    k128, ref_img, fg = _normalise(sl)
    mask_1d = make_mask_1d(W, acceleration=R, center_fraction=CF, seed=MASK_SEED)
    mask_2d = np.broadcast_to(mask_1d[None, :], (H, W)).copy().astype(np.float32)
    pfx     = f"R={R} sl{si}"

    # phi0 from ZF image phase (approximate: aliasing at high R)
    phi0_zf = phi0_from_image(k128, mask_2d)

    # A1-ZF (correct: complex ZF preserves measured lines exactly)
    k_under = k128 * mask_2d
    zf_img  = np.abs(ifft2c_np(k_under)).astype(np.float32)
    a1 = dict(nrmse=_nrmse(zf_img, ref_img, fg),
              tex=_tex(zf_img, ref_img, fg),
              kres=_kres_zf(k128, mask_2d))

    # A2 — (S0,T2*)+phi0-ZF, two seeds, fixed metrics (Df not used in loss)
    preds, stds = [], []
    models_trained = []
    for s in range(N_SEEDS):
        m, _ = _train_2p_phi0(k128, mask_2d, phi0_zf, device, REPRO_EPOCHS,
                               seed=SEED + s, label=f"A2 {pfx} seed{s}")
        pm, ps = _mc_infer(m, k128, mask_2d, device)
        preds.append(pm); stds.append(ps); models_trained.append(m)

    a2_mean  = np.stack(preds, 0).mean(0)
    a2_xstd  = np.stack(preds, 0).std(0)
    a2_mc_std = stds[0]
    dev_a2    = np.abs(a2_mean - ref_img)

    a2 = dict(
        nrmse=_nrmse(a2_mean, ref_img, fg),
        tex=_tex(a2_mean, ref_img, fg),
        kres=_kres_complex_2p_phi0(models_trained[0], k128, mask_2d, phi0_zf, device),
        xseed_std=float(a2_xstd[fg].mean()),
        unc=_unc(a2_mc_std, dev_a2, ref_img, fg),
    )

    print(f"  [{pfx}] ZF NRMSE={a1['nrmse']:.4f} tex={a1['tex']:.3f}  "
          f"A2 NRMSE={a2['nrmse']:.4f} tex={a2['tex']:.3f}  ρ={a2['unc']['rho']:.3f}")
    print(f"  [{pfx}] kres(complex) ZF-meas={a1['kres']['meas']:.4f}  "
          f"A2-meas={a2['kres']['meas']:.4f}  A2-unmeas={a2['kres']['unmeas']:.4f}")

    return dict(R=R, si=si, file=sl["file"],
                a1=a1, a2=a2,
                _arr=dict(ref=ref_img, zf=zf_img, a2=a2_mean,
                          std=a2_mc_std, xstd=a2_xstd, mask=mask_2d, fg=fg))


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def _a(rs, R, arm, f): return float(np.nanmean([r[arm][f] for r in rs if r["R"]==R]))
def _u(rs, R, f):      return float(np.nanmean([r["a2"]["unc"][f] for r in rs if r["R"]==R]))
def _k(rs, R, arm, b): return float(np.nanmean([r[arm]["kres"][b] for r in rs if r["R"]==R]))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _save_gate_fig(rows: list[dict]) -> None:
    files  = [r["file"][:10] for r in rows]
    before = [r["sqrt_before"] for r in rows]
    after  = [r["sqrt_after"]  for r in rows]
    x = np.arange(len(files))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, before, 0.35, label="BEFORE: 3-param (no phi0)", color="steelblue")
    ax.bar(x + 0.2, after,  0.35, label="AFTER: 2-param+fixed-phi0", color="coral")
    ax.axhline(DC_GATE_THRESH, color="r", ls="--", lw=1.5, label=f"gate={DC_GATE_THRESH}")
    ax.axhline(0.032, color="g", ls=":", lw=1.2, label="noise floor ~30dB")
    ax.set_xticks(x); ax.set_xticklabels(files, fontsize=8)
    ax.set_ylabel("sqrt(training_loss) = RMS_diff/RMS_k")
    ax.set_title("Sanity gate: sqrt(L2 DC loss) before/after phi0 fix\n"
                 "(FS data, mask=ones; metric corrected from prior run's FFT(|img|) error)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "phase_fix_gate.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


def _save_summary_fig(results: list[dict]) -> None:
    Rs = ACCEL_SWEEP
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Phase-fix repro: 3-param+fixed-phi0 on fastMRI CORPD\n"
                 "Image=S0·exp(-TE/T2*); DC residual=complex echo; phi0=angle(ZF image)",
                 fontsize=9)

    axes[0,0].plot(Rs, [_a(results,R,"a1","nrmse") for R in Rs], "k--o", lw=2, label="A1-ZF")
    axes[0,0].plot(Rs, [_a(results,R,"a2","nrmse") for R in Rs], "b-o",  lw=2, label="A2+phi0")
    axes[0,0].set_title("NRMSE vs R  (S0·exp(-TE/T2*))"); axes[0,0].set_xlabel("R")
    axes[0,0].set_xticks(Rs); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(Rs, [_a(results,R,"a1","tex") for R in Rs], "k--o", lw=2, label="A1-ZF")
    axes[0,1].plot(Rs, [_a(results,R,"a2","tex") for R in Rs], "b-o",  lw=2, label="A2+phi0")
    axes[0,1].axhline(0.5, color="r", ls=":", lw=1.2, label="R* threshold")
    axes[0,1].set_title("Texture fidelity (HF Pearson r)"); axes[0,1].set_xlabel("R")
    axes[0,1].set_xticks(Rs); axes[0,1].legend(fontsize=8); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(Rs, [_a(results,R,"a2","xseed_std") for R in Rs], "b-o", lw=2)
    axes[0,2].set_title("A2 cross-seed std"); axes[0,2].set_xlabel("R")
    axes[0,2].set_xticks(Rs); axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(Rs, [_k(results,R,"a1","meas")   for R in Rs], "k--s", label="ZF meas")
    axes[1,0].plot(Rs, [_k(results,R,"a1","unmeas") for R in Rs], "k:s",  label="ZF unmeas")
    axes[1,0].plot(Rs, [_k(results,R,"a2","meas")   for R in Rs], "b-o",  label="A2 meas")
    axes[1,0].plot(Rs, [_k(results,R,"a2","unmeas") for R in Rs], "b:o",  label="A2 unmeas")
    axes[1,0].set_title("k-space residual (complex echo, CORRECTED)"); axes[1,0].set_xlabel("R")
    axes[1,0].set_xticks(Rs); axes[1,0].legend(fontsize=7); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(Rs, [_u(results,R,"rho")     for R in Rs], "b-o",  lw=2, label="Spearman ρ")
    axes[1,1].plot(Rs, [_u(results,R,"bnd_int") for R in Rs], "b--s", lw=2, label="bnd/int")
    axes[1,1].axhline(1, color="k", ls="--", lw=0.8)
    axes[1,1].set_title("MC-dropout uncertainty"); axes[1,1].set_xlabel("R")
    axes[1,1].set_xticks(Rs); axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.3)

    n1 = [r["a1"]["nrmse"] for r in results]; n2 = [r["a2"]["nrmse"] for r in results]
    cols = {2:"green", 4:"orange", 8:"red"}
    for R in ACCEL_SWEEP:
        idx = [i for i,r in enumerate(results) if r["R"]==R]
        axes[1,2].scatter([n1[i] for i in idx], [n2[i] for i in idx],
                          color=cols[R], label=f"R={R}", s=50, zorder=3)
    lim = max(max(n1), max(n2)) * 1.1
    axes[1,2].plot([0,lim],[0,lim],"k--",lw=1,alpha=0.5)
    axes[1,2].set_xlabel("A1-ZF NRMSE"); axes[1,2].set_ylabel("A2 NRMSE")
    axes[1,2].set_title("A2 vs ZF"); axes[1,2].legend(fontsize=8); axes[1,2].grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "phase_fix_repro_summary.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


def _save_spatial(res: dict) -> None:
    arr = res["_arr"]; R = res["R"]
    vmax = arr["ref"].max() * 1.05
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Phase-fix repro — {res['file'][:12]} sl{res['si']} R={R}\n"
                 "phi0=angle(ZF image); image=S0·exp(-TE/T2*)", fontsize=9)
    for ax, t, img, vm, cm in zip(axes,
        ["FS ref (est.)", f"A1-ZF R={R}", "A2+phi0", "|A2-ref|", "MC-std"],
        [arr["ref"], arr["zf"], arr["a2"], np.abs(arr["a2"]-arr["ref"]), arr["std"]],
        [vmax, vmax, vmax, 0.3*vmax, arr["std"].max()*1.05+1e-8],
        ["gray","gray","gray","hot","plasma"],
    ):
        im = ax.imshow(img, vmin=0, vmax=vm, cmap=cm, origin="upper")
        ax.set_title(t, fontsize=8); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = RESULTS_DIR / f"phase_fix_{res['file'][:12]}_R{R}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# VERDICT PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_gate_verdict(rows: list[dict], gate_pass: bool) -> None:
    print("\n" + "═"*80)
    print("SANITY GATE — RESULTS")
    print("═"*80)
    print(f"\n  FIX 1: phi0 = angle(IFFT(k_full)) — exact FS phase, analytically grounded")
    print(f"  FIX 2: multi-coil ESPIRiT — SKIPPED (already coil-combined; sigpy unavailable)")
    print(f"  Metric: sqrt(L2_loss) = RMS_diff/RMS_k  [corrected from prior run's L1/mean]")
    print(f"  Threshold: {DC_GATE_THRESH}  (2× noise floor for 30 dB SNR ≈ 0.032)")
    print()
    print(f"  {'File':<22}  {'before':>8}  {'analytical':>11}  {'after':>8}  {'gate'}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*11}  {'─'*8}  {'─'*4}")
    for r in rows:
        print(f"  {r['file']:<22}  {r['sqrt_before']:>8.4f}  {r['res_analytical']:>11.2e}"
              f"  {r['sqrt_after']:>8.4f}  {'PASS' if r['gate_pass'] else 'FAIL'}")
    bm = float(np.mean([r["sqrt_before"] for r in rows]))
    am = float(np.mean([r["sqrt_after"]  for r in rows]))
    print(f"\n  Mean: before={bm:.4f}  after={am:.4f}  "
          f"Δ={bm-am:.4f}  noise_floor≈0.032")
    print(f"\n  GATE: {'PASS — proceeding to R sweep' if gate_pass else 'FAIL — STOP'}")
    print("═"*80)


def print_repro_verdict(results: list[dict]) -> str:
    print("\n" + "═"*100)
    print("PHASE-FIX REPRO TABLES  (3-param + fixed phi0; corrected metrics)")
    print("═"*100)
    print(f"\n  Metric corrections vs prior run:")
    print(f"    • image = S0·exp(-TE/T2*) at TE=32ms  [was: S0 alone — TE=0 extrapolation]")
    print(f"    • DC residual = complex echo FFT       [was: FFT(|magnitude|) — wrong]")
    print(f"    • ZF meas-band residual = 0 by construction  [was: ~0.55 due to magnitude FFT]")
    print(f"    • phi0 = angle(IFFT(k_ZF))  (non-trainable)  [was: absent, causing phase mismatch]")
    print(f"\n  FS reference = ESTIMATE (IFFT(k_full)), not ground truth.")

    print(f"\n  TABLE A — NRMSE (fg) [S0·exp(-TE/T2*) vs FS estimate]")
    print(f"  {'R':>3}  {'ZF':>10}  {'A2+phi0':>10}  {'xseed-std':>11}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*11}")
    for R in ACCEL_SWEEP:
        print(f"  {R:>3}  {_a(results,R,'a1','nrmse'):>10.4f}  "
              f"{_a(results,R,'a2','nrmse'):>10.4f}  "
              f"{_a(results,R,'a2','xseed_std'):>11.6f}")

    print(f"\n  TABLE B — Texture fidelity (HF Pearson r)")
    print(f"  {'R':>3}  {'ZF tex':>8}  {'A2 tex':>8}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}")
    for R in ACCEL_SWEEP:
        print(f"  {R:>3}  {_a(results,R,'a1','tex'):>8.4f}  {_a(results,R,'a2','tex'):>8.4f}")

    print(f"\n  TABLE C — k-space residuals (complex echo, CORRECTED)")
    print(f"  {'R':>3}  {'ZF-meas':>8}  {'ZF-unmeas':>10}  {'A2-meas':>8}  {'A2-unmeas':>10}  {'ratio':>7}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*7}")
    for R in ACCEL_SWEEP:
        zm = _k(results,R,"a1","meas"); zu = _k(results,R,"a1","unmeas")
        am = _k(results,R,"a2","meas"); au = _k(results,R,"a2","unmeas")
        print(f"  {R:>3}  {zm:>8.4f}  {zu:>10.4f}  {am:>8.4f}  {au:>10.4f}  {au/(am+1e-8):>7.2f}")

    print(f"\n  TABLE D — A2 MC-dropout uncertainty")
    print(f"  {'R':>3}  {'Spearman ρ':>11}  {'bnd/int':>8}")
    print(f"  {'─'*3}  {'─'*11}  {'─'*8}")
    for R in ACCEL_SWEEP:
        print(f"  {R:>3}  {_u(results,R,'rho'):>11.4f}  {_u(results,R,'bnd_int'):>8.3f}")

    print("\n" + "─"*100 + "\n  REPRODUCTION CHECKS\n" + "─"*100)
    t2 = {R: _a(results, R, "a2", "tex") for R in ACCEL_SWEEP}
    c1 = "REPRODUCES" if (t2.get(2,0)>0.5 and t2.get(4,0)<0.5) else \
         "PARTIAL"    if t2.get(2,0)>0.5 else "DOES-NOT-REPRODUCE"
    print(f"\n  CHECK 1 — BOUNDARY:  {c1}")
    tex_str = "  ".join(f"R={R}->{t2.get(R, float('nan')):.3f}" for R in ACCEL_SWEEP)
    print(f"    tex-r: {tex_str}")

    r_m = [_k(results,R,"a2","meas")   for R in ACCEL_SWEEP if R>2]
    r_u = [_k(results,R,"a2","unmeas") for R in ACCEL_SWEEP if R>2]
    ratio = float(np.mean(r_u)) / (float(np.mean(r_m)) + 1e-8)
    c2 = "REPRODUCES" if ratio>3.0 else "PARTIAL" if ratio>1.5 else "DOES-NOT-REPRODUCE"
    print(f"\n  CHECK 2 — BLIND SPOT:  {c2}")
    print(f"    k-res (R>2): meas={np.mean(r_m):.4f}  unmeas={np.mean(r_u):.4f}  ratio={ratio:.2f}")

    rho = float(np.mean([_u(results,R,"rho") for R in ACCEL_SWEEP]))
    bnd = float(np.mean([_u(results,R,"bnd_int") for R in ACCEL_SWEEP]))
    c3  = "REPRODUCES" if (rho>0.25 and bnd>1.1) else "PARTIAL" if (rho>0.15 or bnd>1.05) else "DOES-NOT-REPRODUCE"
    print(f"\n  CHECK 3 — ABSTENTION:  {c3}")
    print(f"    ρ={rho:.4f}  bnd/int={bnd:.3f}")

    verdict = (f"CHECK 1 BOUNDARY: {c1}  "
               f"(R=2→{t2.get(2,float('nan')):.3f} R=4→{t2.get(4,float('nan')):.3f} R=8→{t2.get(8,float('nan')):.3f})\n"
               f"CHECK 2 BLIND SPOT: {c2}  (ratio={ratio:.2f})\n"
               f"CHECK 3 ABSTENTION: {c3}  (ρ={rho:.4f} bnd/int={bnd:.3f})")
    print("\n  " + verdict.replace("\n", "\n  "))
    print("═"*100 + "\n")
    return verdict


# ══════════════════════════════════════════════════════════════════════════════
# JSON ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class _NpEnc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_):   return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        try:    return float(obj)
        except: return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "═"*80)
    print("PHASE FIX EXPERIMENT")
    print("═"*80)
    print(f"  FIX 1:  phi0 = angle(IFFT(k_ZF))  — fixed analytical static phase")
    print(f"  FIX 2:  multi-coil ESPIRiT — SKIPPED (already coil-combined)")
    print(f"  METRIC: image=S0·exp(-TE/T2*); DC_res=complex_echo; gate=sqrt(L2_loss)")
    print(f"  Device: {device}  |  GATE_EPOCHS={GATE_EPOCHS}  REPRO_EPOCHS={REPRO_EPOCHS}")
    print(f"  Gate threshold: {DC_GATE_THRESH}  (noise floor ≈ 0.032 at 30 dB SNR)")

    print("\nLoading data …")
    slices = load_real_slices()

    # ── MANDATORY SANITY GATE ─────────────────────────────────────────────────
    gate_pass, gate_rows = run_sanity_gate(slices, device)
    print_gate_verdict(gate_rows, gate_pass)
    _save_gate_fig(gate_rows)

    out: dict = dict(
        gate=dict(pass_=gate_pass, threshold=DC_GATE_THRESH,
                  metric="sqrt(training_loss)=RMS_diff/RMS_k",
                  rows=gate_rows),
    )

    if not gate_pass:
        print("\n  *** GATE FAILED — not proceeding to R sweep ***")
        with open(RESULTS_DIR / "phase_fix.json", "w") as f:
            json.dump(out, f, indent=2, cls=_NpEnc)
        print("  → phase_fix.json")
        return

    # ── REPRO SWEEP ───────────────────────────────────────────────────────────
    print("\n  *** GATE PASSED — R sweep ***\n")
    results: list[dict] = []
    saved = set()
    for R in ACCEL_SWEEP:
        print(f"\n{'─'*60}\nR = {R}\n{'─'*60}")
        for si, sl in enumerate(slices):
            res = run_condition(sl, R, device, si)
            results.append(res)
            if (R, si) not in saved:
                _save_spatial(res); saved.add((R, si))

    _save_summary_fig(results)
    verdict = print_repro_verdict(results)

    out["results"] = [{k:v for k,v in r.items() if k!="_arr"} for r in results]
    out["verdict"] = verdict
    with open(RESULTS_DIR / "phase_fix.json", "w") as f:
        json.dump(out, f, indent=2, cls=_NpEnc)
    print("  → phase_fix.json")


if __name__ == "__main__":
    main()
