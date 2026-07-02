#!/usr/bin/env python3
"""
OpenNeuro ds006926 — Two-Branch Multi-Echo Identifiability Repro
=================================================================
DATASET: ds006926 "Complex multi-echo fMRI", Siemens 3T, BIDS.
  10 subjects (sub-a01..sub-a10). Volumes: 64×64×40 voxels, 3×3×3.5 mm.

BRANCH A — Anat MEGRE magnitude (10 echoes, TEs≈2.5..80 ms):
  Signal: S(TE) = S0·exp(-TE/T2*)  — magnitude ONLY, NO Df, NO phi0.
  Df and phi0 are NOT present here; magnitude MEGRE carries only T2* information.

BRANCH B — Func complex sbref (3 echoes, TEs≈12/29/46 ms):
  Signal: S(TE) = S0·exp(-TE/T2*)·exp(i·(2π·Df·TE + phi0))
  KEEPS BOTH phi0 (static per-voxel phase) AND Df (off-resonance Hz).
  3 echoes separate phi0 (constant-in-TE) from Df·TE (linear-in-TE).
  LIMITATION: ΔTE≈17 ms → unambiguous Df range ≈ ±29 Hz. Voxels with
              |Df|>29 Hz will have aliased phase-slope estimates.

BONUS: Correlate recovered Branch-B Df with fmap phasediff → Hz.
  Siemens old dcm2niix phase encoding: raw ∈ [-4096, 4094] → radians
    phase_rad = raw × π / 4096
  fmap → Hz: field_Hz = phasediff_rad / (2π × ΔTE_fmap)

SANITY GATE (mandatory, blocking, ANALYTICAL — no network training):
  Branch A: sqrt(kres_2p_analytic) < GATE_A_THRESH  (log-linear OLS)
  Branch B: sqrt(kres_4p_analytic) < GATE_B_THRESH
            AND kres_4p / kres_2p < GATE_B_RATIO
  HARD STOP if either branch fails.

ARMS per branch:
  A1: Zero-filled (ZF) reconstruction
  A2: Physics-grounded CNN, self-supervised k-space DC loss
  A3: Black-box CNN, supervised on analytical labels (sub-a01..sub-a08);
      tested on sub-a09..sub-a10

R sweep: {2, 4, 8}  |  1D random mask, CF=0.08, MASK_SEED=42

DOWNLOAD: S3 REST API (public, no credentials).
  URL: https://s3.amazonaws.com/openneuro.org/{dataset}/{path}
  TEs ALWAYS read from JSON sidecars — never hardcoded.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel not found — run: pip install nibabel")
try:
    import requests as _requests
except ImportError:
    sys.exit("requests not found — run: pip install requests")
try:
    from skimage.restoration import unwrap_phase as _skimage_unwrap
    HAVE_SKIMAGE = True
except ImportError:
    _skimage_unwrap = None
    HAVE_SKIMAGE = False

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
_GATE_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _GATE_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_GATE_DIR))

from run_gate import fft2c_np, ifft2c_np, make_mask_1d, RESULTS_DIR   # noqa: E402
from run_calib_uncertainty import BottleneckNet3MC                       # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATASET    = "ds006926"
S3_BASE    = "https://s3.amazonaws.com/openneuro.org"
SUBJECTS   = [f"sub-a{i:02d}" for i in range(1, 11)]
TRAIN_SUBS = SUBJECTS[:8]    # sub-a01..sub-a08 — A3 training
TEST_SUBS  = SUBJECTS[8:]    # sub-a09..sub-a10 — gate + repro test

N_ECHOES_A = 10              # anat MEGRE
N_ECHOES_B = 3               # func sbref
SLICE_Z    = 20              # central z-slice (40 slices total, 0-indexed)
H = W      = 64              # confirmed 64×64 in-plane for this dataset

# Siemens old dcm2niix: phase raw ∈ [-4096, 4094] → radians
SIEMENS_PHASE_SCALE = math.pi / 4096.0

ACCEL_SWEEP  = [2, 4, 8]
CF           = 0.08
MASK_SEED    = 42
REPRO_EPOCHS = 150
A3_EPOCHS    = 250
TRAIN_LR     = 5e-4
HIDDEN       = 32
DROPOUT_P    = 0.10
N_MC         = 20
N_SEEDS      = 2
SEED         = 42
HF_SIGMA     = 2.0

# Gate thresholds (analytical, FS data, real brain noise)
GATE_A_THRESH = 0.25   # 2-param log-linear OLS; anat MEGRE is clean magnitude
# Branch B: EPI sbref has geometric distortion + multiexponential T2* → absolute
# residual ~0.6-0.7 is expected even with correct physics. Primary check is the
# RATIO (4-param must outperform 2-param), not the absolute value.
GATE_B_THRESH = 0.80   # absolute 4-param kres — loose upper bound for real EPI
GATE_B_RATIO  = 0.90   # kres_4p / kres_2p must be < 0.90 (Df must help)

T2_MIN, T2_MAX = 5.0, 150.0
DF_BOUND       = 150.0       # ±150 Hz Df clamp in network output

DATA_DIR = _REPO_ROOT / "data" / DATASET

# ─────────────────────────────────────────────────────────────────────────────
#  DOWNLOAD UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_TOTAL_BYTES: int = 0


def _s3_get(key: str, dest: Path, session: _requests.Session) -> int:
    """Download one S3 object; returns bytes fetched (0 = cached)."""
    global _TOTAL_BYTES
    if dest.exists():
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BASE}/{key}"
    r = session.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {url}")
    n = 0
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65536):
            fh.write(chunk); n += len(chunk)
    _TOTAL_BYTES += n
    return n


def _anat_pairs(sub: str) -> list[tuple[str, str]]:
    pairs = []
    for e in range(1, N_ECHOES_A + 1):
        stem = f"{sub}_echo-{e}_MEGRE"
        base = f"{DATASET}/{sub}/anat/{stem}"
        pairs += [(f"{base}.nii.gz", f"{stem}.nii.gz"),
                  (f"{base}.json",   f"{stem}.json")]
    return pairs


def _func_pairs(sub: str) -> list[tuple[str, str]]:
    task = "task-rest_acq-tr1800"
    pairs = []
    for e in range(1, N_ECHOES_B + 1):
        pfx  = f"{sub}_{task}_echo-{e}"
        base = f"{DATASET}/{sub}/func"
        # JSON sidecar in this dataset includes part- entity (non-standard but confirmed)
        json_stem = f"{pfx}_part-mag_sbref.json"
        pairs.append((f"{base}/{json_stem}", json_stem))
        for part in ("mag", "phase"):
            stem = f"{pfx}_part-{part}_sbref"
            pairs.append((f"{base}/{stem}.nii.gz", f"{stem}.nii.gz"))
    return pairs


def _fmap_pairs(sub: str) -> list[tuple[str, str]]:
    base = f"{DATASET}/{sub}/fmap"
    return [
        (f"{base}/{sub}_phasediff.nii.gz", f"{sub}_phasediff.nii.gz"),
        (f"{base}/{sub}_phasediff.json",   f"{sub}_phasediff.json"),
    ]


def download_all(session: _requests.Session) -> None:
    print("\n─── DOWNLOAD (S3 REST API, no credentials) ───")
    for sub in SUBJECTS:
        sub_dir = DATA_DIR / sub
        pairs   = _anat_pairs(sub) + _func_pairs(sub) + _fmap_pairs(sub)
        for key, stem in pairs:
            n = _s3_get(key, sub_dir / stem, session)
            if n > 0:
                print(f"  ↓ {sub}/{stem}  ({n/1e6:.2f} MB)")
    gb = _TOTAL_BYTES / 1e9
    print(f"  Total downloaded this run: {gb:.3f} GB  "
          f"({_TOTAL_BYTES:,} bytes)")


# ─────────────────────────────────────────────────────────────────────────────
#  VERIFICATION (HARD STOP on error)
# ─────────────────────────────────────────────────────────────────────────────

def verify_download() -> None:
    print("\n─── DOWNLOAD VERIFICATION ───")
    for sub in SUBJECTS:
        sub_dir = DATA_DIR / sub
        # Anat MEGRE: exactly N_ECHOES_A nii.gz + N_ECHOES_A json
        anat_niis = sorted(sub_dir.glob(f"{sub}_echo-*_MEGRE.nii.gz"))
        if len(anat_niis) != N_ECHOES_A:
            sys.exit(f"HARD STOP: {sub} anat MEGRE — "
                     f"expected {N_ECHOES_A} echoes, found {len(anat_niis)}")
        # Func sbref: part-phase must exist for each echo
        task = "task-rest_acq-tr1800"
        for e in range(1, N_ECHOES_B + 1):
            phase_f = sub_dir / f"{sub}_{task}_echo-{e}_part-phase_sbref.nii.gz"
            if not phase_f.exists():
                sys.exit(f"HARD STOP: {sub} func phase echo-{e} missing")
        # Parse and print TEs to confirm JSON access
        te_a = _parse_tes_anat(sub)
        te_b = _parse_tes_func(sub)
        print(f"  {sub}: anat-TEs=[{te_a[0]:.2f}..{te_a[-1]:.2f}] ms  "
              f"func-TEs={[f'{t:.2f}' for t in te_b]} ms")
    print("  PASS\n")


# ─────────────────────────────────────────────────────────────────────────────
#  JSON SIDECAR PARSING (TEs always from JSON — never hardcoded)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tes_anat(sub: str) -> list[float]:
    """Return anat MEGRE TEs in ms, from per-echo JSON sidecars."""
    tes = []
    for e in range(1, N_ECHOES_A + 1):
        jp = DATA_DIR / sub / f"{sub}_echo-{e}_MEGRE.json"
        with open(jp) as fh:
            d = json.load(fh)
        tes.append(float(d["EchoTime"]) * 1000.0)   # s → ms
    return tes


def _parse_tes_func(sub: str) -> list[float]:
    """Return func sbref TEs in ms. JSON includes part-mag in filename (dataset-specific)."""
    tes = []
    task = "task-rest_acq-tr1800"
    for e in range(1, N_ECHOES_B + 1):
        jp = DATA_DIR / sub / f"{sub}_{task}_echo-{e}_part-mag_sbref.json"
        with open(jp) as fh:
            d = json.load(fh)
        tes.append(float(d["EchoTime"]) * 1000.0)
    return tes


def _parse_fmap_dte(sub: str) -> float:
    """Return fmap ΔTE = EchoTime2 - EchoTime1 in seconds."""
    jp = DATA_DIR / sub / f"{sub}_phasediff.json"
    with open(jp) as fh:
        d = json.load(fh)
    return float(d["EchoTime2"]) - float(d["EchoTime1"])


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _nii_slice(path: Path) -> np.ndarray:
    """Load 3D NIfTI and extract central z-slice → [H, W] float32."""
    arr = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    return arr[:, :, SLICE_Z]


def load_branch_a(sub: str) -> tuple[np.ndarray, list[float]]:
    """
    Returns:
        echoes: [N_ECHOES_A, H, W] float32 magnitude images
        TEs_ms: echo times from JSON sidecars
    """
    TEs_ms = _parse_tes_anat(sub)
    echoes = []
    for e in range(1, N_ECHOES_A + 1):
        echoes.append(_nii_slice(DATA_DIR / sub / f"{sub}_echo-{e}_MEGRE.nii.gz"))
    return np.stack(echoes, 0), TEs_ms


def load_branch_b(sub: str) -> tuple[np.ndarray, list[float]]:
    """
    Returns:
        echoes_cpx: [N_ECHOES_B, H, W] complex64  (mag × exp(i·phase_rad))
                    Siemens phase scaling: phase_rad = raw / 4096 × π
        TEs_ms: echo times from JSON sidecars
    """
    TEs_ms = _parse_tes_func(sub)
    task   = "task-rest_acq-tr1800"
    echoes = []
    for e in range(1, N_ECHOES_B + 1):
        mag   = _nii_slice(DATA_DIR / sub / f"{sub}_{task}_echo-{e}_part-mag_sbref.nii.gz")
        raw_p = _nii_slice(DATA_DIR / sub / f"{sub}_{task}_echo-{e}_part-phase_sbref.nii.gz")
        phase = raw_p * SIEMENS_PHASE_SCALE    # → radians ∈ [-π, π]
        echoes.append((mag * np.cos(phase) + 1j * mag * np.sin(phase)).astype(np.complex64))
    return np.stack(echoes, 0), TEs_ms


def load_fmap_hz(sub: str) -> tuple[np.ndarray, float]:
    """
    Returns:
        field_Hz: [H, W] float32 off-resonance map in Hz
        dte_s: ΔTE in seconds used for conversion
    """
    dte_s  = _parse_fmap_dte(sub)
    raw_pd = _nii_slice(DATA_DIR / sub / f"{sub}_phasediff.nii.gz")
    pd_rad = raw_pd * SIEMENS_PHASE_SCALE
    if HAVE_SKIMAGE:
        pd_rad = _skimage_unwrap(pd_rad.astype(np.float64)).astype(np.float32)
    field_Hz = (pd_rad / (2.0 * math.pi * dte_s)).astype(np.float32)
    return field_Hz, dte_s


# ─────────────────────────────────────────────────────────────────────────────
#  FFT TORCH
# ─────────────────────────────────────────────────────────────────────────────

def fft2c_torch(x: torch.Tensor) -> torch.Tensor:
    # norm="ortho" matches fft2c_np convention; omitting it causes 64× amplitude mismatch
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYTICAL FITTING (GATE — zero training epochs)
# ─────────────────────────────────────────────────────────────────────────────

def fit_t2star_ols(
    echoes: np.ndarray,     # [N_ECHOES, H, W] magnitude
    TEs_ms: list[float],
    fg_thresh: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Log-linear OLS per voxel: log(S_e) = log(S0) + (1/T2*) × (-TE_e).
    Returns: s0_map, t2_map [H, W] float32, fg_mask [H, W] bool
    """
    TEs = np.array(TEs_ms, dtype=np.float64)
    A   = np.column_stack([np.ones(len(TEs)), -TEs])           # [E, 2]
    ATA_inv = np.linalg.inv(A.T @ A + 1e-10 * np.eye(2))
    Proj    = ATA_inv @ A.T                                     # [2, E]

    fg = echoes[0] > fg_thresh * (echoes[0].max() + 1e-8)
    s0_map = np.zeros((H, W), np.float32)
    t2_map = np.full((H, W), T2_MAX, np.float32)

    vals    = echoes[:, fg].astype(np.float64)                  # [E, N_fg]
    log_v   = np.log(np.maximum(vals, 1e-6))
    coeffs  = Proj @ log_v                                      # [2, N_fg]
    s0_map[fg] = np.exp(np.clip(coeffs[0], -10, 10)).astype(np.float32)
    t2_map[fg] = np.clip(1.0 / (coeffs[1] + 1e-9), T2_MIN, T2_MAX).astype(np.float32)
    return s0_map, t2_map, fg


def fit_df_phase_ols(
    echoes_cpx: np.ndarray,   # [N_ECHOES_B, H, W] complex64
    TEs_ms: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical 4-param fit for complex multi-echo data.

    1. phi0 = angle(echo[0])  — static receive/background phase
    2. Df from inter-echo phase-slope OLS relative to echo-0
       LIMITATION: unambiguous range ≈ ±1/(2·ΔTE_min) Hz (≈ ±29 Hz here)
    3. S0, T2* from log-linear OLS on magnitude echoes

    Returns: s0_map, t2_map, df_map, phi0  — all [H, W] float32
    """
    TEs_s = np.array(TEs_ms, dtype=np.float64) / 1000.0
    phi0  = np.angle(echoes_cpx[0]).astype(np.float32)

    # Spatially unwrap each echo's phase
    phases = []
    for e in range(N_ECHOES_B):
        p = np.angle(echoes_cpx[e]).astype(np.float64)
        if HAVE_SKIMAGE:
            p = _skimage_unwrap(p)
        phases.append(p)
    phases = np.array(phases)   # [N_ECHOES_B, H, W]

    # Inter-echo phase difference: Δφ(e) = φ(TE_e) - φ(TE_0) = 2π·Df·(TE_e - TE_0)
    delta_phases = phases - phases[0:1]                              # [N_B, H, W]
    delta_TEs    = TEs_s - TEs_s[0]                                  # [N_B]
    # Use echoes e≥1 (echo-0 is trivially zero)
    dTE  = delta_TEs[1:]                                             # [N_B-1]
    dPhi = delta_phases[1:]                                          # [N_B-1, H, W]
    denom     = float(np.sum(dTE**2)) + 1e-15
    numerator = np.tensordot(dTE, dPhi, axes=([0], [0]))             # [H, W]
    df_map    = (numerator / denom / (2.0 * math.pi)).astype(np.float32)

    mag_echoes = np.abs(echoes_cpx)
    s0_map, t2_map, _ = fit_t2star_ols(mag_echoes, TEs_ms)
    return s0_map, t2_map, df_map, phi0


def _kres_2p_np(
    s0: np.ndarray, t2: np.ndarray,
    k_echoes: np.ndarray,    # [N, H, W] complex — k-space of magnitude OR complex echoes
    TEs_ms: list[float],
    mask_2d: np.ndarray,
    use_mag: bool = True,    # True: k-space of mag; False: complex echo k-space
) -> float:
    """Relative k-space RMS residual for 2-param model (no Df, no phi0)."""
    total = norm = 0.0
    for e, te in enumerate(TEs_ms):
        mag = s0 * np.exp(-te / (t2 + 1e-4))
        if use_mag:
            k_pred = fft2c_np(mag.astype(np.complex64))
        else:
            k_pred = fft2c_np(mag.astype(np.complex64))   # phase=0
        k_m  = k_echoes[e] * mask_2d
        diff = mask_2d * (k_pred - k_m)
        total += float(np.sum(np.abs(diff)**2))
        norm  += float(np.sum(np.abs(k_m)**2))
    return math.sqrt(total / (norm + 1e-12))


def _kres_4p_np(
    s0: np.ndarray, t2: np.ndarray, df: np.ndarray, phi0: np.ndarray,
    k_echoes_cpx: np.ndarray,   # [N_ECHOES_B, H, W] complex k-space
    TEs_ms: list[float],
    mask_2d: np.ndarray,
) -> float:
    """Relative k-space RMS residual for 4-param complex model."""
    total = norm = 0.0
    for e, te_ms in enumerate(TEs_ms):
        te_s = te_ms / 1000.0
        mag  = s0 * np.exp(-te_ms / (t2 + 1e-4))
        ph   = 2.0 * math.pi * df * te_s + phi0
        cpx  = (mag * np.cos(ph) + 1j * mag * np.sin(ph)).astype(np.complex64)
        k_pred = fft2c_np(cpx)
        k_m    = k_echoes_cpx[e] * mask_2d
        diff   = mask_2d * (k_pred - k_m)
        total += float(np.sum(np.abs(diff)**2))
        norm  += float(np.sum(np.abs(k_m)**2))
    return math.sqrt(total / (norm + 1e-12))


def run_gate_a(sub: str) -> dict:
    """Branch A gate: 2-param log-linear OLS residual < GATE_A_THRESH."""
    echoes, TEs_ms = load_branch_a(sub)
    sc = float(echoes[0].max()) + 1e-8
    echoes = echoes / sc
    mask_fs = np.ones((H, W), np.float32)
    # k-space of magnitude images (what the retrospective undersampling simulates)
    k_full = np.stack([fft2c_np(e.astype(np.complex64)) for e in echoes], 0)
    s0, t2, fg = fit_t2star_ols(echoes, TEs_ms)
    kres = _kres_2p_np(s0, t2, k_full, TEs_ms, mask_fs, use_mag=True)
    gate_pass = kres < GATE_A_THRESH
    t2_mean = float(t2[fg].mean()) if fg.sum() > 0 else float("nan")
    print(f"  Gate-A {sub}: kres_2p={kres:.4f}  T2*_mean={t2_mean:.1f}ms  "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return dict(sub=sub, kres_2p=kres, t2_mean=t2_mean,
                gate_thresh=GATE_A_THRESH, gate_pass=gate_pass)


def run_gate_b(sub: str) -> dict:
    """Branch B gate: 4-param complex residual < GATE_B_THRESH AND ratio < GATE_B_RATIO."""
    echoes_cpx, TEs_ms = load_branch_b(sub)
    sc = float(np.abs(echoes_cpx[0]).max()) + 1e-8
    echoes_cpx = echoes_cpx / sc
    mask_fs = np.ones((H, W), np.float32)
    k_full  = np.stack([fft2c_np(echoes_cpx[e]) for e in range(N_ECHOES_B)], 0)
    s0, t2, df, phi0 = fit_df_phase_ols(echoes_cpx, TEs_ms)
    fg = np.abs(echoes_cpx[0]) > 0.05 * np.abs(echoes_cpx[0]).max()
    kres_2p = _kres_2p_np(s0, t2, k_full, TEs_ms, mask_fs, use_mag=False)
    kres_4p = _kres_4p_np(s0, t2, df, phi0, k_full, TEs_ms, mask_fs)
    ratio   = kres_4p / (kres_2p + 1e-10)
    gate_pass = (kres_4p < GATE_B_THRESH) and (ratio < GATE_B_RATIO)
    df_mean = float(df[fg].mean()) if fg.sum() > 0 else float("nan")
    print(f"  Gate-B {sub}: kres_2p={kres_2p:.4f}  kres_4p={kres_4p:.4f}  "
          f"ratio={ratio:.3f}  Df_mean={df_mean:.1f}Hz  "
          f"{'PASS' if gate_pass else 'FAIL'}")
    return dict(sub=sub, kres_2p=kres_2p, kres_4p=kres_4p, ratio=ratio,
                df_mean=df_mean, gate_thresh_kres=GATE_B_THRESH,
                gate_thresh_ratio=GATE_B_RATIO, gate_pass=gate_pass)


# ─────────────────────────────────────────────────────────────────────────────
#  NETWORK ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

class MagNet(nn.Module):
    """5-block 2D CNN for magnitude-only T2* fitting (Branch A)."""

    def __init__(self, n_echoes: int = N_ECHOES_A, hidden: int = HIDDEN,
                 T2_min: float = T2_MIN, T2_max: float = T2_MAX,
                 p: float = DROPOUT_P) -> None:
        super().__init__()
        self.T2_min = T2_min; self.T2_max = T2_max
        G = min(8, hidden)   # GroupNorm groups — prevents activation explosion across 5 blocks
        self.enc = nn.Sequential(
            nn.Conv2d(n_echoes, hidden, 3, padding=1), nn.GroupNorm(G, hidden), nn.ReLU(),
            nn.Conv2d(hidden,   hidden, 3, padding=1), nn.GroupNorm(G, hidden), nn.ReLU(), nn.Dropout2d(p),
            nn.Conv2d(hidden,   hidden, 3, padding=1), nn.GroupNorm(G, hidden), nn.ReLU(), nn.Dropout2d(p),
            nn.Conv2d(hidden,   hidden, 3, padding=1), nn.GroupNorm(G, hidden), nn.ReLU(), nn.Dropout2d(p),
            nn.Conv2d(hidden,   hidden, 3, padding=1), nn.GroupNorm(G, hidden), nn.ReLU(),
        )
        self.s0_head = nn.Conv2d(hidden, 1, 1)
        self.t2_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.t2_head.weight)
        # Start T2* at 30ms (true brain mean); sigmoid_inv((30-T2_min)/(T2_max-30)) ≈ -1.57
        nn.init.constant_(self.t2_head.bias, math.log((30.0 - T2_min) / (T2_max - 30.0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h  = self.enc(x)
        s0 = F.softplus(self.s0_head(h))
        t2 = self.T2_min + (self.T2_max - self.T2_min) * torch.sigmoid(self.t2_head(h))
        return s0, t2


# ─────────────────────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def kspace_loss_mag(
    s0: torch.Tensor, t2: torch.Tensor,
    k_tc: torch.Tensor,     # [N_ECHOES_A, H, W] complex — k-space of magnitude echoes
    mask_tc: torch.Tensor,
    TEs_ms: list[float],
) -> torch.Tensor:
    """
    Branch A self-supervised DC loss on k-space of magnitude images.
    Per-echo normalization: each echo contributes equally regardless of signal level.
    This prevents the S0→0 trivial minimum caused by late-echo dominance in joint norm.
    """
    s0_ = s0[0, 0]; t2_ = t2[0, 0]
    loss = torch.zeros(1, device=s0.device)
    for e, te in enumerate(TEs_ms):
        mag_pred = s0_ * torch.exp(-te / t2_)
        cpx_pred = torch.view_as_complex(
            torch.stack([mag_pred, torch.zeros_like(mag_pred)], -1).contiguous()
        )
        k_pred  = fft2c_torch(cpx_pred)
        k_m     = k_tc[e]
        dr      = mask_tc * (k_pred.real - k_m.real)
        di      = mask_tc * (k_pred.imag - k_m.imag)
        e_norm  = (mask_tc * k_m.abs()**2).sum() + 1e-8
        loss    = loss + (dr**2 + di**2).sum() / e_norm
    return loss / len(TEs_ms)


def kspace_loss_cpx(
    s0: torch.Tensor, t2: torch.Tensor, df: torch.Tensor,
    phi0_tc: torch.Tensor,   # [H, W] fixed (non-trainable)
    k_tc: torch.Tensor,      # [N_ECHOES_B, H, W] complex
    mask_tc: torch.Tensor,
    TEs_ms: list[float],
) -> torch.Tensor:
    """
    Branch B self-supervised DC loss: S(TE) = S0·exp(-TE/T2*)·exp(i·(2π·Df·TE+phi0)).
    Per-echo normalization prevents Df blowup from making loss >> 1 and masking S0 gradient.
    """
    s0_ = s0[0, 0]; t2_ = t2[0, 0]; df_ = df[0, 0]
    loss = torch.zeros(1, device=s0.device)
    for e, te_ms in enumerate(TEs_ms):
        te_s  = te_ms / 1000.0
        mag   = s0_ * torch.exp(-te_ms / t2_)
        phase = 2.0 * math.pi * df_ * te_s + phi0_tc
        cpx   = torch.view_as_complex(
            torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], -1).contiguous()
        )
        k_pred  = fft2c_torch(cpx)
        k_m     = k_tc[e]
        dr      = mask_tc * (k_pred.real - k_m.real)
        di      = mask_tc * (k_pred.imag - k_m.imag)
        e_norm  = (mask_tc * k_m.abs()**2).sum() + 1e-8
        loss    = loss + (dr**2 + di**2).sum() / e_norm
    return loss / len(TEs_ms)


# ─────────────────────────────────────────────────────────────────────────────
#  ZF INPUT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _zf_input_mag(k_echoes: np.ndarray) -> np.ndarray:
    """[N_ECHOES_A, H, W] float32 — magnitude of ZF images per echo."""
    return np.stack([np.abs(ifft2c_np(k_echoes[e])).astype(np.float32)
                     for e in range(k_echoes.shape[0])], 0)


def _zf_input_cpx(k_echoes_cpx: np.ndarray) -> np.ndarray:
    """[2×N_ECHOES_B, H, W] float32 — real/imag channels of ZF complex images."""
    channels = []
    for e in range(k_echoes_cpx.shape[0]):
        img = ifft2c_np(k_echoes_cpx[e])
        channels += [img.real.astype(np.float32), img.imag.astype(np.float32)]
    return np.stack(channels, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _train_magnet(
    k_full: np.ndarray, mask_2d: np.ndarray,
    TEs_ms: list[float], device: torch.device,
    n_epochs: int, seed: int = SEED, label: str = "",
) -> tuple[MagNet, float]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    k_under = k_full * mask_2d[np.newaxis]
    zf_in   = torch.from_numpy(_zf_input_mag(k_under)).unsqueeze(0).to(device)
    k_tc    = torch.from_numpy(k_under).to(torch.complex64).to(device)
    mask_tc = torch.from_numpy(mask_2d).to(device)
    model   = MagNet(n_echoes=N_ECHOES_A, hidden=HIDDEN, T2_min=T2_MIN,
                     T2_max=T2_MAX, p=DROPOUT_P).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=TRAIN_LR * 0.01)
    log = max(1, n_epochs // 4); final = float("nan")
    for ep in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()
        s0, t2 = model(zf_in)
        loss   = kspace_loss_mag(s0, t2, k_tc, mask_tc, TEs_ms)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log == 0 or ep == n_epochs:
            final = float(loss.item())
            if label: print(f"    [{label}] ep {ep:4d}/{n_epochs}  loss={final:.5f}")
    return model, final


def _train_cpxnet(
    k_full_cpx: np.ndarray, mask_2d: np.ndarray,
    TEs_ms: list[float], phi0: np.ndarray,
    device: torch.device, n_epochs: int,
    seed: int = SEED, label: str = "",
) -> tuple[BottleneckNet3MC, float]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    k_under  = k_full_cpx * mask_2d[np.newaxis]
    zf_in    = torch.from_numpy(_zf_input_cpx(k_under)).unsqueeze(0).to(device)
    k_tc     = torch.from_numpy(k_under).to(torch.complex64).to(device)
    mask_tc  = torch.from_numpy(mask_2d).to(device)
    phi0_tc  = torch.from_numpy(phi0).to(device)
    model    = BottleneckNet3MC(n_echoes=N_ECHOES_B, hidden=HIDDEN, T2_min=T2_MIN,
                                T2_max=T2_MAX, df_bound=DF_BOUND, p=DROPOUT_P).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=TRAIN_LR * 0.01)
    log = max(1, n_epochs // 4); final = float("nan")
    for ep in range(1, n_epochs + 1):
        model.train(); opt.zero_grad()
        s0, t2, df = model(zf_in)
        loss = kspace_loss_cpx(s0, t2, df, phi0_tc, k_tc, mask_tc, TEs_ms)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log == 0 or ep == n_epochs:
            final = float(loss.item())
            if label: print(f"    [{label}] ep {ep:4d}/{n_epochs}  loss={final:.5f}")
    return model, final


def _train_a3_mag(
    train_subs: list[str], device: torch.device, n_epochs: int,
) -> MagNet:
    """Supervised A3 for Branch A: fit T2* maps from 8 training subjects."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = MagNet(n_echoes=N_ECHOES_A, hidden=HIDDEN, T2_min=T2_MIN,
                   T2_max=T2_MAX, p=DROPOUT_P).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=TRAIN_LR * 0.01)
    # Pre-compute: ZF inputs + analytical labels (FS → no undersampling for training)
    inputs_tc, s0_tcs, t2_tcs = [], [], []
    for sub in train_subs:
        echoes, TEs_ms = load_branch_a(sub)
        sc = float(echoes[0].max()) + 1e-8; echoes = echoes / sc
        k_full  = np.stack([fft2c_np(e.astype(np.complex64)) for e in echoes], 0)
        zf      = _zf_input_mag(k_full)    # FS ZF = FS image (mask=ones)
        s0, t2, _ = fit_t2star_ols(echoes, TEs_ms)
        inputs_tc.append(torch.from_numpy(zf).unsqueeze(0).to(device))
        s0_tcs.append(torch.from_numpy(s0).unsqueeze(0).unsqueeze(0).to(device))
        t2_tcs.append(torch.from_numpy(t2).unsqueeze(0).unsqueeze(0).to(device))
    N = len(inputs_tc)
    log = max(1, n_epochs // 4)
    for ep in range(1, n_epochs + 1):
        idx = (ep - 1) % N
        model.train(); opt.zero_grad()
        s0_pred, t2_pred = model(inputs_tc[idx])
        loss = (F.mse_loss(s0_pred / (s0_tcs[idx].max() + 1e-8),
                           s0_tcs[idx] / (s0_tcs[idx].max() + 1e-8)) +
                F.mse_loss(t2_pred / T2_MAX, t2_tcs[idx] / T2_MAX))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log == 0 or ep == n_epochs:
            print(f"    [A3-A] ep {ep:4d}/{n_epochs}  loss={float(loss.item()):.5f}")
    return model


def _train_a3_cpx(
    train_subs: list[str], device: torch.device, n_epochs: int,
) -> BottleneckNet3MC:
    """Supervised A3 for Branch B: fit (T2*, Df) maps from training subjects."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = BottleneckNet3MC(n_echoes=N_ECHOES_B, hidden=HIDDEN, T2_min=T2_MIN,
                             T2_max=T2_MAX, df_bound=DF_BOUND, p=DROPOUT_P).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_epochs, eta_min=TRAIN_LR * 0.01)
    inputs_tc, t2_tcs, df_tcs = [], [], []
    for sub in train_subs:
        echoes_cpx, TEs_ms = load_branch_b(sub)
        sc = float(np.abs(echoes_cpx[0]).max()) + 1e-8; echoes_cpx = echoes_cpx / sc
        k_full = np.stack([fft2c_np(echoes_cpx[e]) for e in range(N_ECHOES_B)], 0)
        zf     = _zf_input_cpx(k_full)
        _, t2, df, _ = fit_df_phase_ols(echoes_cpx, TEs_ms)
        inputs_tc.append(torch.from_numpy(zf).unsqueeze(0).to(device))
        t2_tcs.append(torch.from_numpy(t2).unsqueeze(0).unsqueeze(0).to(device))
        df_tcs.append(torch.from_numpy(df).unsqueeze(0).unsqueeze(0).to(device))
    N = len(inputs_tc)
    log = max(1, n_epochs // 4)
    for ep in range(1, n_epochs + 1):
        idx = (ep - 1) % N
        model.train(); opt.zero_grad()
        _, t2_pred, df_pred = model(inputs_tc[idx])
        loss = (F.mse_loss(t2_pred / T2_MAX, t2_tcs[idx] / T2_MAX) +
                F.mse_loss(df_pred / DF_BOUND, df_tcs[idx] / DF_BOUND))
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log == 0 or ep == n_epochs:
            print(f"    [A3-B] ep {ep:4d}/{n_epochs}  loss={float(loss.item()):.5f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _nrmse(pred: np.ndarray, ref: np.ndarray, fg: np.ndarray) -> float:
    return float(np.linalg.norm(pred[fg] - ref[fg]) / (np.linalg.norm(ref[fg]) + 1e-8))


def _tex(pred: np.ndarray, ref: np.ndarray, fg: np.ndarray) -> float:
    hp = pred - gaussian_filter(pred.astype(np.float64), HF_SIGMA)
    hr = ref  - gaussian_filter(ref.astype(np.float64),  HF_SIGMA)
    if fg.sum() < 20: return float("nan")
    r, _ = pearsonr(hp[fg], hr[fg])
    return float(r) if np.isfinite(r) else 0.0


def _unc(std: np.ndarray, dev: np.ndarray, fg: np.ndarray) -> float:
    """Spearman ρ between MC-std and absolute error."""
    if fg.sum() < 20: return 0.0
    rho, _ = spearmanr(std[fg], dev[fg])
    return float(rho) if np.isfinite(rho) else 0.0


def _kres_net_mag(
    model: MagNet, k_full: np.ndarray, mask_2d: np.ndarray,
    TEs_ms: list[float], device: torch.device,
) -> dict:
    """k-space residuals from MagNet (for blind-spot check)."""
    k_under = k_full * mask_2d[np.newaxis]
    zf_in   = torch.from_numpy(_zf_input_mag(k_under)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        s0, t2 = model(zf_in)
    s0_ = s0[0, 0].cpu().numpy(); t2_ = t2[0, 0].cpu().numpy()
    meas = mask_2d.astype(bool)
    m_tot = u_tot = r_tot = 0.0
    for e, te in enumerate(TEs_ms):
        mag_pred = s0_ * np.exp(-te / (t2_ + 1e-4))
        k_pred   = fft2c_np(mag_pred.astype(np.complex64))
        diff     = np.abs(k_pred - k_full[e])
        ref_m    = np.abs(k_full[e])
        m_tot  += float(diff[meas].mean() / (ref_m[meas].mean() + 1e-10))
        u_tot  += float(diff[~meas].mean() / (ref_m[~meas].mean() + 1e-10))
    N = len(TEs_ms)
    return dict(meas=m_tot/N, unmeas=u_tot/N)


def _kres_net_cpx(
    model: BottleneckNet3MC, k_full: np.ndarray, phi0: np.ndarray,
    mask_2d: np.ndarray, TEs_ms: list[float], device: torch.device,
) -> dict:
    """k-space residuals from CpxNet (Branch B blind-spot check)."""
    k_under = k_full * mask_2d[np.newaxis]
    zf_in   = torch.from_numpy(_zf_input_cpx(k_under)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        s0, t2, df = model(zf_in)
    s0_ = s0[0,0].cpu().numpy(); t2_ = t2[0,0].cpu().numpy(); df_ = df[0,0].cpu().numpy()
    meas = mask_2d.astype(bool)
    m_tot = u_tot = 0.0
    for e, te_ms in enumerate(TEs_ms):
        te_s = te_ms / 1000.0
        mag  = s0_ * np.exp(-te_ms / (t2_ + 1e-4))
        ph   = 2.0 * math.pi * df_ * te_s + phi0
        cpx  = (mag * np.cos(ph) + 1j * mag * np.sin(ph)).astype(np.complex64)
        k_pred = fft2c_np(cpx)
        diff   = np.abs(k_pred - k_full[e])
        ref_m  = np.abs(k_full[e])
        m_tot += float(diff[meas].mean() / (ref_m[meas].mean() + 1e-10))
        u_tot += float(diff[~meas].mean() / (ref_m[~meas].mean() + 1e-10))
    N = len(TEs_ms)
    return dict(meas=m_tot/N, unmeas=u_tot/N)


# ─────────────────────────────────────────────────────────────────────────────
#  BRANCH A — REPRO CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_branch_a_condition(
    sub: str, R: int,
    a3_model: MagNet,
    device: torch.device,
) -> dict:
    echoes, TEs_ms = load_branch_a(sub)
    sc = float(echoes[0].max()) + 1e-8; echoes = echoes / sc
    k_full  = np.stack([fft2c_np(e.astype(np.complex64)) for e in echoes], 0)
    mask_1d = make_mask_1d(W, acceleration=R, center_fraction=CF, seed=MASK_SEED)
    mask_2d = np.broadcast_to(mask_1d[None, :], (H, W)).copy().astype(np.float32)
    ref_img = echoes[0]
    fg      = ref_img > 0.05 * ref_img.max()

    # A1: ZF of echo-1 under mask
    k_under_e1 = k_full[0] * mask_2d
    zf_img     = np.abs(ifft2c_np(k_under_e1)).astype(np.float32)
    meas       = mask_2d.astype(bool)
    a1_kres    = dict(
        meas=float(np.abs(fft2c_np(zf_img.astype(np.complex64)) - k_full[0])[meas].mean() /
                   (np.abs(k_full[0])[meas].mean() + 1e-10)),
        unmeas=float(np.abs(fft2c_np(zf_img.astype(np.complex64)) - k_full[0])[~meas].mean() /
                     (np.abs(k_full[0])[~meas].mean() + 1e-10)),
    )
    a1 = dict(nrmse=_nrmse(zf_img, ref_img, fg), tex=_tex(zf_img, ref_img, fg), kres=a1_kres)

    # A2: MagNet — two seeds, averaged
    t2_samples_all, s0_mean_all = [], []
    a2_kres_list = []
    for s in range(N_SEEDS):
        m, _ = _train_magnet(k_full, mask_2d, TEs_ms, device, REPRO_EPOCHS,
                              seed=SEED + s, label=f"A2-A {sub} R={R} s{s}")
        # MC-dropout T2* samples
        k_under = k_full * mask_2d[np.newaxis]
        zf_in   = torch.from_numpy(_zf_input_mag(k_under)).unsqueeze(0).to(device)
        m.train()
        t2_mc = []
        with torch.no_grad():
            for _ in range(N_MC):
                _, t2 = m(zf_in); t2_mc.append(t2[0,0].cpu().numpy())
        t2_samples_all.append(np.stack(t2_mc, 0).mean(0))
        # S0·exp(-TE1/T2*) as reconstruction image
        m.eval()
        with torch.no_grad():
            s0_t, t2_t = m(zf_in)
        s0_n = s0_t[0,0].cpu().numpy(); t2_n = t2_t[0,0].cpu().numpy()
        pred_img = s0_n * np.exp(-TEs_ms[0] / (t2_n + 1e-4))
        s0_mean_all.append(pred_img)
        a2_kres_list.append(_kres_net_mag(m, k_full, mask_2d, TEs_ms, device))

    a2_pred  = np.stack(s0_mean_all, 0).mean(0).astype(np.float32)
    t2_mean  = np.stack(t2_samples_all, 0).mean(0).astype(np.float32)
    t2_std   = np.stack(t2_samples_all, 0).std(0).astype(np.float32)
    dev_a2   = np.abs(a2_pred - ref_img)
    a2_kres  = dict(meas=float(np.mean([k["meas"] for k in a2_kres_list])),
                    unmeas=float(np.mean([k["unmeas"] for k in a2_kres_list])))
    a2 = dict(nrmse=_nrmse(a2_pred, ref_img, fg), tex=_tex(a2_pred, ref_img, fg),
              kres=a2_kres, unc=_unc(t2_std, dev_a2, fg))

    # A3: supervised model (pre-trained on training subjects)
    k_under = k_full * mask_2d[np.newaxis]
    zf_in   = torch.from_numpy(_zf_input_mag(k_under)).unsqueeze(0).to(device)
    a3_model.eval()
    with torch.no_grad():
        s0_a3, t2_a3 = a3_model(zf_in)
    s0_a3 = s0_a3[0,0].cpu().numpy(); t2_a3 = t2_a3[0,0].cpu().numpy()
    pred_a3 = s0_a3 * np.exp(-TEs_ms[0] / (t2_a3 + 1e-4))
    a3 = dict(nrmse=_nrmse(pred_a3, ref_img, fg), tex=_tex(pred_a3, ref_img, fg),
              kres=_kres_net_mag(a3_model, k_full, mask_2d, TEs_ms, device))

    print(f"  A-{sub} R={R}: ZF NRMSE={a1['nrmse']:.4f}  A2={a2['nrmse']:.4f}  A3={a3['nrmse']:.4f}  "
          f"A2-unmeas={a2['kres']['unmeas']:.4f}  ρ={a2['unc']:.3f}")
    return dict(sub=sub, R=R, a1=a1, a2=a2, a3=a3,
                _arr=dict(ref=ref_img, zf=zf_img, a2=a2_pred, t2=t2_mean,
                          t2_std=t2_std, mask=mask_2d, fg=fg))


# ─────────────────────────────────────────────────────────────────────────────
#  BRANCH B — REPRO CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_branch_b_condition(
    sub: str, R: int,
    a3_model: BottleneckNet3MC,
    device: torch.device,
) -> dict:
    echoes_cpx, TEs_ms = load_branch_b(sub)
    sc = float(np.abs(echoes_cpx[0]).max()) + 1e-8; echoes_cpx = echoes_cpx / sc
    k_full  = np.stack([fft2c_np(echoes_cpx[e]) for e in range(N_ECHOES_B)], 0)
    mask_1d = make_mask_1d(W, acceleration=R, center_fraction=CF, seed=MASK_SEED)
    mask_2d = np.broadcast_to(mask_1d[None, :], (H, W)).copy().astype(np.float32)
    # phi0 from ZF of undersampled echo-1
    k_under_e1 = k_full[0] * mask_2d
    phi0_zf    = np.angle(ifft2c_np(k_under_e1)).astype(np.float32)
    ref_mag    = np.abs(ifft2c_np(k_full[0])).astype(np.float32)
    fg         = ref_mag > 0.05 * ref_mag.max()

    # A1: ZF magnitude of echo-1
    zf_img = np.abs(ifft2c_np(k_under_e1)).astype(np.float32)
    meas   = mask_2d.astype(bool)
    diff0  = np.abs(fft2c_np(zf_img.astype(np.complex64)) - k_full[0])
    ref_m0 = np.abs(k_full[0])
    a1_kres = dict(meas=float(diff0[meas].mean() / (ref_m0[meas].mean() + 1e-10)),
                   unmeas=float(diff0[~meas].mean() / (ref_m0[~meas].mean() + 1e-10)))
    a1 = dict(nrmse=_nrmse(zf_img, ref_mag, fg), tex=_tex(zf_img, ref_mag, fg), kres=a1_kres)

    # A2: CpxNet — two seeds
    pred_a2_list, df_mc_list, a2_kres_list = [], [], []
    for s in range(N_SEEDS):
        m, _ = _train_cpxnet(k_full, mask_2d, TEs_ms, phi0_zf, device, REPRO_EPOCHS,
                              seed=SEED + s, label=f"A2-B {sub} R={R} s{s}")
        k_under = k_full * mask_2d[np.newaxis]
        zf_in   = torch.from_numpy(_zf_input_cpx(k_under)).unsqueeze(0).to(device)
        m.train()
        df_mc = []
        with torch.no_grad():
            for _ in range(N_MC):
                _, _, df = m(zf_in); df_mc.append(df[0,0].cpu().numpy())
        df_mc_list.append(np.stack(df_mc, 0))
        m.eval()
        with torch.no_grad():
            s0_t, t2_t, _ = m(zf_in)
        pred_a2_list.append((s0_t[0,0].cpu().numpy()
                             * np.exp(-TEs_ms[0] / (t2_t[0,0].cpu().numpy() + 1e-4))))
        a2_kres_list.append(_kres_net_cpx(m, k_full, phi0_zf, mask_2d, TEs_ms, device))

    a2_pred  = np.stack(pred_a2_list, 0).mean(0).astype(np.float32)
    df_mean  = np.concatenate(df_mc_list, 0).mean(0).astype(np.float32)
    df_std   = np.concatenate(df_mc_list, 0).std(0).astype(np.float32)
    dev_a2   = np.abs(a2_pred - ref_mag)
    a2_kres  = dict(meas=float(np.mean([k["meas"] for k in a2_kres_list])),
                    unmeas=float(np.mean([k["unmeas"] for k in a2_kres_list])))
    a2 = dict(nrmse=_nrmse(a2_pred, ref_mag, fg), tex=_tex(a2_pred, ref_mag, fg),
              kres=a2_kres, unc=_unc(df_std, dev_a2, fg),
              df_mean=float(df_mean[fg].mean()) if fg.sum() > 0 else float("nan"))

    # A3
    k_under = k_full * mask_2d[np.newaxis]
    zf_in   = torch.from_numpy(_zf_input_cpx(k_under)).unsqueeze(0).to(device)
    a3_model.eval()
    with torch.no_grad():
        s0_a3, t2_a3, _ = a3_model(zf_in)
    pred_a3 = (s0_a3[0,0].cpu().numpy()
               * np.exp(-TEs_ms[0] / (t2_a3[0,0].cpu().numpy() + 1e-4)))
    a3 = dict(nrmse=_nrmse(pred_a3, ref_mag, fg), tex=_tex(pred_a3, ref_mag, fg),
              kres=_kres_net_cpx(a3_model, k_full, phi0_zf, mask_2d, TEs_ms, device))

    print(f"  B-{sub} R={R}: ZF NRMSE={a1['nrmse']:.4f}  A2={a2['nrmse']:.4f}  A3={a3['nrmse']:.4f}  "
          f"A2-unmeas={a2['kres']['unmeas']:.4f}  ρ={a2['unc']:.3f}  "
          f"Df_mean={a2['df_mean']:.1f}Hz")
    return dict(sub=sub, R=R, a1=a1, a2=a2, a3=a3,
                _arr=dict(ref=ref_mag, zf=zf_img, a2=a2_pred,
                          df=df_mean, df_std=df_std, mask=mask_2d, fg=fg))


# ─────────────────────────────────────────────────────────────────────────────
#  BONUS — fmap correlation
# ─────────────────────────────────────────────────────────────────────────────

def bonus_fmap_correlation(
    sub: str, R: int,
    a3_model_b: BottleneckNet3MC,
    device: torch.device,
) -> Optional[dict]:
    """
    Correlate recovered Df (Branch B A2) with fmap phasediff → Hz.
    Both maps are on the same 64×64 grid (same voxel size); no registration needed.
    LIMITATION: fmap ΔTE=2.46ms → max unambiguous field = ±203Hz.
                Branch-B ΔTE≈17ms → max unambiguous field = ±29Hz.
    """
    try:
        field_Hz, dte_s = load_fmap_hz(sub)
    except Exception as e:
        print(f"  BONUS fmap: SKIP ({e})")
        return None

    echoes_cpx, TEs_ms = load_branch_b(sub)
    sc = float(np.abs(echoes_cpx[0]).max()) + 1e-8; echoes_cpx = echoes_cpx / sc
    k_full  = np.stack([fft2c_np(echoes_cpx[e]) for e in range(N_ECHOES_B)], 0)
    mask_1d = make_mask_1d(W, acceleration=R, center_fraction=CF, seed=MASK_SEED)
    mask_2d = np.broadcast_to(mask_1d[None, :], (H, W)).copy().astype(np.float32)
    k_under_e1 = k_full[0] * mask_2d
    phi0_zf    = np.angle(ifft2c_np(k_under_e1)).astype(np.float32)
    fg         = np.abs(ifft2c_np(k_full[0])) > 0.05 * np.abs(ifft2c_np(k_full[0])).max()

    # A2 Df map
    m, _ = _train_cpxnet(k_full, mask_2d, TEs_ms, phi0_zf, device, REPRO_EPOCHS,
                          label=f"Bonus-B {sub} R={R}")
    k_under = k_full * mask_2d[np.newaxis]
    zf_in   = torch.from_numpy(_zf_input_cpx(k_under)).unsqueeze(0).to(device)
    m.eval()
    with torch.no_grad():
        _, _, df_t = m(zf_in)
    df_a2 = df_t[0,0].cpu().numpy().astype(np.float32)

    # Analytical Df
    _, _, df_ana, _ = fit_df_phase_ols(echoes_cpx, TEs_ms)

    # Correlations in foreground (brain tissue mask)
    fmap_fg    = field_Hz[fg]
    df_a2_fg   = df_a2[fg]
    df_ana_fg  = df_ana[fg]

    rho_a2,  _ = spearmanr(df_a2_fg, fmap_fg)
    rho_ana, _ = spearmanr(df_ana_fg, fmap_fg)
    r_a2,    _ = pearsonr(df_a2_fg, fmap_fg)
    r_ana,   _ = pearsonr(df_ana_fg, fmap_fg)

    print(f"  BONUS {sub} R={R}: Spearman(A2-Df, fmap)={rho_a2:.3f}  "
          f"Spearman(ana-Df, fmap)={rho_ana:.3f}  "
          f"Pearson(A2-Df, fmap)={r_a2:.3f}")
    return dict(sub=sub, R=R, rho_a2=float(rho_a2), rho_ana=float(rho_ana),
                r_a2=float(r_a2), r_ana=float(r_ana), dte_s=dte_s,
                _arr=dict(df_a2=df_a2, df_ana=df_ana, fmap=field_Hz, fg=fg))


# ─────────────────────────────────────────────────────────────────────────────
#  TABLES + VERDICT
# ─────────────────────────────────────────────────────────────────────────────

def _a(rs: list[dict], R: int, arm: str, f: str) -> float:
    vals = [r[arm][f] for r in rs if r["R"] == R and f in r[arm]]
    return float(np.nanmean(vals)) if vals else float("nan")

def _k(rs: list[dict], R: int, arm: str, band: str) -> float:
    vals = [r[arm]["kres"][band] for r in rs if r["R"] == R]
    return float(np.nanmean(vals)) if vals else float("nan")

def _u(rs: list[dict], R: int) -> float:
    vals = [r["a2"]["unc"] for r in rs if r["R"] == R]
    return float(np.nanmean(vals)) if vals else float("nan")


def print_gate_table(gate_rows_a: list[dict], gate_rows_b: list[dict]) -> tuple[bool, bool]:
    print("\n" + "═"*80)
    print("SANITY GATE RESULTS")
    print("═"*80)

    print("\n  BRANCH A (magnitude, log-linear OLS):")
    print(f"  {'Subject':<12}  {'kres_2p':>8}  {'T2*_mean':>10}  {'gate'}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*4}")
    for r in gate_rows_a:
        print(f"  {r['sub']:<12}  {r['kres_2p']:>8.4f}  {r['t2_mean']:>9.1f}ms"
              f"  {'PASS' if r['gate_pass'] else 'FAIL'}")
    gate_a = all(r["gate_pass"] for r in gate_rows_a)
    print(f"  Threshold: sqrt(kres_2p) < {GATE_A_THRESH}  →  BRANCH A: {'PASS' if gate_a else 'FAIL'}")

    print(f"\n  BRANCH B (complex, 4-param OLS):")
    if not HAVE_SKIMAGE:
        print("  WARNING: skimage unavailable — phase unwrapping SKIPPED; "
              "Df estimates may be noisy")
    print(f"  {'Subject':<12}  {'kres_2p':>8}  {'kres_4p':>8}  {'ratio':>7}  "
          f"{'Df_mean':>9}  {'gate'}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*4}")
    for r in gate_rows_b:
        print(f"  {r['sub']:<12}  {r['kres_2p']:>8.4f}  {r['kres_4p']:>8.4f}  "
              f"{r['ratio']:>7.3f}  {r['df_mean']:>8.1f}Hz  "
              f"{'PASS' if r['gate_pass'] else 'FAIL'}")
    gate_b = all(r["gate_pass"] for r in gate_rows_b)
    print(f"  Thresholds: kres_4p < {GATE_B_THRESH}, ratio < {GATE_B_RATIO}"
          f"  →  BRANCH B: {'PASS' if gate_b else 'FAIL'}")
    print("  LIMITATION (Branch B): ΔTE≈17ms → unambiguous Df ≈ ±29Hz; "
          "voxels with |Df|>29Hz alias.")
    print("═"*80)
    return gate_a, gate_b


def print_branch_tables(label: str, results: list[dict]) -> str:
    print(f"\n{'═'*90}")
    print(f"BRANCH {label} — REPRO TABLES")
    print(f"{'═'*90}")

    print(f"\n  TABLE A — NRMSE (fg) [ref = FS echo-1 magnitude]")
    print(f"  {'R':>3}  {'ZF':>8}  {'A2':>8}  {'A3':>8}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}")
    for R in ACCEL_SWEEP:
        print(f"  {R:>3}  {_a(results,R,'a1','nrmse'):>8.4f}"
              f"  {_a(results,R,'a2','nrmse'):>8.4f}"
              f"  {_a(results,R,'a3','nrmse'):>8.4f}")

    print(f"\n  TABLE B — Texture fidelity (HF Pearson r)")
    print(f"  {'R':>3}  {'ZF':>8}  {'A2':>8}  {'A3':>8}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}")
    for R in ACCEL_SWEEP:
        print(f"  {R:>3}  {_a(results,R,'a1','tex'):>8.4f}"
              f"  {_a(results,R,'a2','tex'):>8.4f}"
              f"  {_a(results,R,'a3','tex'):>8.4f}")

    print(f"\n  TABLE C — k-space residuals (A2)")
    print(f"  {'R':>3}  {'ZF-meas':>9}  {'ZF-unmeas':>11}  "
          f"{'A2-meas':>9}  {'A2-unmeas':>11}  {'ratio':>7}")
    print(f"  {'─'*3}  {'─'*9}  {'─'*11}  {'─'*9}  {'─'*11}  {'─'*7}")
    for R in ACCEL_SWEEP:
        zm = _k(results,R,"a1","meas"); zu = _k(results,R,"a1","unmeas")
        am = _k(results,R,"a2","meas"); au = _k(results,R,"a2","unmeas")
        print(f"  {R:>3}  {zm:>9.4f}  {zu:>11.4f}  {am:>9.4f}  {au:>11.4f}"
              f"  {au/(am+1e-8):>7.2f}")

    print(f"\n  TABLE D — A2 MC-uncertainty Spearman ρ")
    print(f"  {'R':>3}  {'ρ(std,err)':>12}")
    print(f"  {'─'*3}  {'─'*12}")
    for R in ACCEL_SWEEP:
        print(f"  {R:>3}  {_u(results, R):>12.4f}")

    # Reproduction checks
    print(f"\n  ─── REPRO CHECKS ───")
    t2  = {R: _a(results, R, "a2", "tex") for R in ACCEL_SWEEP}
    c1  = ("REPRODUCES"       if (t2.get(2,0)>0.5 and t2.get(4,0)<0.5) else
           "PARTIAL"          if t2.get(2,0)>0.5 else
           "DOES-NOT-REPRODUCE")
    print(f"  CHECK 1 BOUNDARY:  {c1}  "
          f"(tex R=2→{t2.get(2,float('nan')):.3f} "
          f"R=4→{t2.get(4,float('nan')):.3f} "
          f"R=8→{t2.get(8,float('nan')):.3f})")

    r_m = [_k(results,R,"a2","meas")   for R in ACCEL_SWEEP if R > 2]
    r_u = [_k(results,R,"a2","unmeas") for R in ACCEL_SWEEP if R > 2]
    ratio = float(np.mean(r_u)) / (float(np.mean(r_m)) + 1e-8)
    c2   = ("REPRODUCES"       if ratio > 3.0 else
            "PARTIAL"          if ratio > 1.5 else
            "DOES-NOT-REPRODUCE")
    print(f"  CHECK 2 BLIND SPOT: {c2}  "
          f"(unmeas/meas ratio={ratio:.2f} at R>2)")

    rho  = float(np.mean([_u(results, R) for R in ACCEL_SWEEP]))
    c3   = ("REPRODUCES"       if rho > 0.25 else
            "PARTIAL"          if rho > 0.15 else
            "DOES-NOT-REPRODUCE")
    print(f"  CHECK 3 ABSTENTION: {c3}  (mean Spearman ρ={rho:.4f})")
    print(f"{'═'*90}")

    return (f"Branch {label}: "
            f"CHECK1={c1} (R=2→{t2.get(2,float('nan')):.3f})  "
            f"CHECK2={c2} (ratio={ratio:.2f})  "
            f"CHECK3={c3} (ρ={rho:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _save_branch_fig(label: str, results: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    Rs = ACCEL_SWEEP
    fig.suptitle(f"Branch {label} — OpenNeuro ds006926 repro (2 test subjects)", fontsize=10)
    cols = {2: "green", 4: "orange", 8: "red"}

    ax = axes[0, 0]
    ax.plot(Rs, [_a(results,R,"a1","nrmse") for R in Rs], "k--o", label="A1-ZF")
    ax.plot(Rs, [_a(results,R,"a2","nrmse") for R in Rs], "b-o",  label="A2-phys")
    ax.plot(Rs, [_a(results,R,"a3","nrmse") for R in Rs], "r-s",  label="A3-super")
    ax.set_title("NRMSE vs R"); ax.set_xlabel("R"); ax.legend(fontsize=8)
    ax.set_xticks(Rs); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(Rs, [_a(results,R,"a1","tex") for R in Rs], "k--o", label="A1-ZF")
    ax.plot(Rs, [_a(results,R,"a2","tex") for R in Rs], "b-o",  label="A2-phys")
    ax.plot(Rs, [_a(results,R,"a3","tex") for R in Rs], "r-s",  label="A3-super")
    ax.axhline(0.5, color="r", ls=":", lw=1.2, label="R*=0.5")
    ax.set_title("Texture r vs R"); ax.set_xlabel("R"); ax.legend(fontsize=8)
    ax.set_xticks(Rs); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(Rs, [_k(results,R,"a2","meas")   for R in Rs], "b-o",  label="A2 meas")
    ax.plot(Rs, [_k(results,R,"a2","unmeas") for R in Rs], "b:o",  label="A2 unmeas")
    ax.plot(Rs, [_k(results,R,"a1","unmeas") for R in Rs], "k:s",  label="ZF unmeas")
    ax.set_title("k-space residuals"); ax.set_xlabel("R"); ax.legend(fontsize=8)
    ax.set_xticks(Rs); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(Rs, [_u(results, R) for R in Rs], "b-o")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title("MC-uncertainty Spearman ρ"); ax.set_xlabel("R")
    ax.set_xticks(Rs); ax.grid(True, alpha=0.3)

    # Scatter: A2 vs ZF NRMSE coloured by R
    ax = axes[1, 1]
    for R in Rs:
        idxs = [i for i, r in enumerate(results) if r["R"] == R]
        n1 = [results[i]["a1"]["nrmse"] for i in idxs]
        n2 = [results[i]["a2"]["nrmse"] for i in idxs]
        ax.scatter(n1, n2, color=cols[R], label=f"R={R}", s=60, zorder=3)
    lim = max([r["a1"]["nrmse"] for r in results] + [r["a2"]["nrmse"] for r in results]) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("ZF NRMSE"); ax.set_ylabel("A2 NRMSE")
    ax.set_title("A2 vs ZF"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Spatial example: test sub-a09 at R=4
    R_ex = 4
    ex   = next((r for r in results if r["sub"] == TEST_SUBS[0] and r["R"] == R_ex), None)
    ax   = axes[1, 2]
    if ex is not None:
        arr  = ex["_arr"]
        vmax = arr["ref"].max() * 1.05
        ax.imshow(np.hstack([arr["ref"], arr["zf"], arr["a2"]]),
                  vmin=0, vmax=vmax, cmap="gray", origin="upper")
        ax.set_title(f"{TEST_SUBS[0]} R={R_ex}: FS | ZF | A2", fontsize=8)
    ax.axis("off")

    fig.tight_layout()
    out = RESULTS_DIR / f"openneuro_{label.lower()}_repro.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


def _save_fmap_fig(bonus_rows: list[dict]) -> None:
    rows = [r for r in bonus_rows if r is not None]
    if not rows: return
    r0  = rows[0]
    arr = r0["_arr"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"BONUS: fmap ↔ recovered Df  ({r0['sub']})\n"
                 f"Spearman ρ(A2, fmap)={r0['rho_a2']:.3f}  "
                 f"ρ(ana, fmap)={r0['rho_ana']:.3f}", fontsize=9)
    vmax = max(np.abs(arr["fmap"]).max(), np.abs(arr["df_a2"]).max()) * 0.5
    for ax, title, img in zip(axes,
        ["fmap → Hz", "A2 Df (Hz)", "Analytical Df (Hz)"],
        [arr["fmap"], arr["df_a2"], arr["df_ana"]],
    ):
        im = ax.imshow(img, vmin=-vmax, vmax=vmax, cmap="bwr", origin="upper")
        ax.set_title(title, fontsize=9); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = RESULTS_DIR / "openneuro_fmap_correlation.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  JSON ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class _NpEnc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        try: return float(obj)
        except: return super().default(obj)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "═"*80)
    print("OpenNeuro ds006926 — Two-Branch Multi-Echo Identifiability Repro")
    print("═"*80)
    print(f"  Branch A: anat MEGRE magnitude  ({N_ECHOES_A} echoes, no Df, no phi0)")
    print(f"  Branch B: func complex sbref    ({N_ECHOES_B} echoes, Df + phi0 retained)")
    print(f"  Device: {device}  |  REPRO_EPOCHS={REPRO_EPOCHS}  A3_EPOCHS={A3_EPOCHS}")
    print(f"  skimage unwrap: {'YES' if HAVE_SKIMAGE else 'NO (phase unwrapping skipped)'}")
    print(f"  Gate: ANALYTICAL — no network training in gate")

    # ── STEP 0: download ─────────────────────────────────────────────────────
    t0 = time.time()
    with _requests.Session() as sess:
        download_all(sess)
    verify_download()

    # ── SANITY GATE (mandatory, blocking) ────────────────────────────────────
    print("─── SANITY GATE (analytical OLS, test subjects only) ───")
    gate_rows_a = [run_gate_a(sub) for sub in TEST_SUBS]
    gate_rows_b = [run_gate_b(sub) for sub in TEST_SUBS]
    gate_a, gate_b = print_gate_table(gate_rows_a, gate_rows_b)

    if not gate_a:
        sys.exit("HARD STOP: Branch A gate FAILED — "
                 "magnitude k-space residual exceeds threshold. "
                 "Check data download or signal model.")
    if not gate_b:
        sys.exit("HARD STOP: Branch B gate FAILED — "
                 "complex k-space residual or 4p/2p ratio exceeds threshold. "
                 "Check phase rescaling or Df fit.")

    print("\n  *** BOTH GATES PASSED — proceeding to R sweep ***\n")

    # ── A3 TRAINING (cross-subject supervised) ───────────────────────────────
    print("─── A3 Training (sub-a01..sub-a08, supervised) ───")
    print("  Training Branch-A A3 (MagNet) …")
    a3_model_a = _train_a3_mag(TRAIN_SUBS, device, A3_EPOCHS)
    print("  Training Branch-B A3 (BottleneckNet3MC) …")
    a3_model_b = _train_a3_cpx(TRAIN_SUBS, device, A3_EPOCHS)

    # ── R SWEEP ───────────────────────────────────────────────────────────────
    results_a: list[dict] = []
    results_b: list[dict] = []
    bonus_rows: list[Optional[dict]] = []

    for R in ACCEL_SWEEP:
        print(f"\n{'─'*60}\nR = {R}\n{'─'*60}")
        for sub in TEST_SUBS:
            results_a.append(run_branch_a_condition(sub, R, a3_model_a, device))
            results_b.append(run_branch_b_condition(sub, R, a3_model_b, device))

    # ── BONUS fmap (R=4 only for brevity) ────────────────────────────────────
    print("\n─── BONUS: fmap ↔ Df correlation (R=4) ───")
    for sub in TEST_SUBS:
        bonus_rows.append(bonus_fmap_correlation(sub, R=4, a3_model_b=a3_model_b,
                                                  device=device))

    # ── TABLES ───────────────────────────────────────────────────────────────
    verdict_a = print_branch_tables("A (magnitude, 10-echo MEGRE)", results_a)
    verdict_b = print_branch_tables("B (complex, 3-echo sbref)", results_b)

    if any(r is not None for r in bonus_rows):
        valid = [r for r in bonus_rows if r is not None]
        rho_mean = float(np.mean([r["rho_a2"] for r in valid]))
        print(f"\n  BONUS fmap correlation: mean Spearman ρ(A2-Df, fmap) = {rho_mean:.3f}")
        bonus_verdict = (f"CORRELATES" if rho_mean > 0.4 else
                         "WEAK"        if rho_mean > 0.2 else
                         "UNCORRELATED")
        print(f"  BONUS VERDICT: {bonus_verdict} (ρ={rho_mean:.3f})")
        print(f"  LIMITATION: Branch-B ΔTE≈17ms limits unambiguous Df to ±29Hz; "
              f"fmap range wider.")
    else:
        bonus_verdict = "SKIP"

    # ── TOTAL GB pulled ───────────────────────────────────────────────────────
    total_gb = _TOTAL_BYTES / 1e9
    print(f"\n  Total data pulled: {total_gb:.3f} GB  ({_TOTAL_BYTES:,} bytes)")
    print(f"  Total wall time: {(time.time()-t0)/60:.1f} min")

    # ── FIGURES ───────────────────────────────────────────────────────────────
    _save_branch_fig("A", results_a)
    _save_branch_fig("B", results_b)
    _save_fmap_fig(bonus_rows)

    # ── JSON DUMP ─────────────────────────────────────────────────────────────
    out_dict = dict(
        dataset=DATASET, n_subjects=len(SUBJECTS), train_subs=TRAIN_SUBS,
        test_subs=TEST_SUBS, total_gb_pulled=total_gb,
        gate=dict(branch_a=gate_rows_a, branch_b=gate_rows_b,
                  branch_a_pass=gate_a, branch_b_pass=gate_b),
        branch_a=[{k: v for k, v in r.items() if k != "_arr"} for r in results_a],
        branch_b=[{k: v for k, v in r.items() if k != "_arr"} for r in results_b],
        bonus_fmap=[{k: v for k, v in r.items() if k != "_arr"}
                    for r in bonus_rows if r is not None],
        verdict_a=verdict_a, verdict_b=verdict_b, bonus_verdict=bonus_verdict,
        limitations=[
            "Branch A: k-space is retrospectively simulated from magnitude NIfTI "
            "(actual complex acquisition k-space unavailable from magnitude-only files).",
            "Branch B: ΔTE≈17ms → unambiguous Df ≈ ±29Hz; higher Df aliases in "
            "phase-slope OLS. No temporal unwrapping applied.",
            "A3 trained on FS data (mask=ones), tested on undersampled — "
            "distribution shift between train and test inputs.",
            "Reference image = FS echo-1 magnitude (estimate, not GT).",
        ],
    )
    out_path = RESULTS_DIR / "openneuro_repro.json"
    with open(out_path, "w") as fh:
        json.dump(out_dict, fh, indent=2, cls=_NpEnc)
    print(f"  → {out_path.name}")

    print("\n" + "═"*80)
    print("SUMMARY")
    print("═"*80)
    print(f"  {verdict_a}")
    print(f"  {verdict_b}")
    print(f"  BONUS: {bonus_verdict}")
    print(f"  Data: {total_gb:.3f} GB pulled from ds006926 (S3 REST API)")
    print("═"*80 + "\n")


if __name__ == "__main__":
    main()
