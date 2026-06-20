#!/usr/bin/env python3
"""
Identifiability Gate Experiment
================================
Answers: does a bottleneck network recover TRUE tissue parameters when trained
only on MR signal-model k-space consistency (no ground-truth labels)?

PART 1  Generate 128×128 Shepp-Logan-style phantom with known S0 / T2* maps.
        Synthesise multi-echo images at TEs=[5,15,25,35,45] ms.
        FFT to k-space (same centred convention as the reconstruction pipeline).
        Undersample at 4× acceleration, 8% centre fraction.
        Add complex Gaussian k-space noise at configurable SNR.

PART 2  BottleneckNet: [n_echoes, H, W] ZF-magnitude → S0_hat [H,W],
        T2*_hat [H,W] (via softplus / sigmoid).
        Training signal: k-space data-consistency loss ONLY (no GT maps).
        Supervised variant uses direct MSE on S0 + T2* maps as sanity check.

PART 3  Four methods evaluated on the same phantom:
          1. Analytical log-linear fit  — fully-sampled echoes  (upper bound)
          2. Analytical log-linear fit  — zero-filled echoes    (naive baseline)
          3. Physics-only bottleneck    — k-space loss, 3 seeds (real test)
          4. Supervised bottleneck      — GT-map loss           (arch sanity check)
        Cross-seed T2* std across seeds probes identifiability.
        Figures saved to experiments/identifiability_gate/results/.

Usage:
    python experiments/identifiability_gate/run_gate.py
    python experiments/identifiability_gate/run_gate.py --epochs 600 --snr 25
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────── paths / reproducibility ─────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

H, W       = 128, 128
TEs_MS     = [5.0, 15.0, 25.0, 35.0, 45.0]   # echo times in milliseconds
N_ECHOES   = len(TEs_MS)
ACCEL      = 4
CF         = 0.08   # centre fraction
N_SEEDS    = 3
HIDDEN     = 48     # CNN channel width
T2_MIN     = 5.0    # ms — lower bound on predicted T2*
T2_MAX     = 150.0  # ms — upper bound on predicted T2*


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — PHANTOM + FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def make_phantom(H: int = 128, W: int = 128):
    """
    Five-region Shepp-Logan-style brain phantom.

    Regions (in override order — later masks win):
        background  : T2*=5 ms, S0=0.05
        gray matter : T2*=55 ms, S0=1.00   (outer ellipse)
        white matter: T2*=25 ms, S0=0.70   (inner ellipse)
        CSF         : T2*=75 ms, S0=0.60   (two lateral-ventricle ellipses)
        lesion      : T2*=35 ms, S0=0.85   (small circle, off-centre)

    Returns S0_map [H,W], T2s_map [H,W], labels [H,W] (int, 0=background).
    """
    yy, xx = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]  # normalised coords ∈ [-1, 1]

    S0  = np.full((H, W), 0.05, dtype=np.float32)
    T2s = np.full((H, W), 5.0,  dtype=np.float32)
    lbl = np.zeros((H, W), dtype=np.int32)

    def ellipse(cx, cy, rx, ry):
        return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0

    # Gray matter
    gm = ellipse(0, 0, 0.80, 0.90)
    S0[gm] = 1.00; T2s[gm] = 55.0; lbl[gm] = 1

    # White matter
    wm = ellipse(0, 0, 0.50, 0.65)
    S0[wm] = 0.70; T2s[wm] = 25.0; lbl[wm] = 2

    # CSF — left lateral ventricle
    csf_l = ellipse(-0.22, 0, 0.10, 0.25)
    S0[csf_l] = 0.60; T2s[csf_l] = 75.0; lbl[csf_l] = 3

    # CSF — right lateral ventricle
    csf_r = ellipse(+0.22, 0, 0.10, 0.25)
    S0[csf_r] = 0.60; T2s[csf_r] = 75.0; lbl[csf_r] = 3

    # Lesion
    lesion = ellipse(0.38, -0.20, 0.08, 0.08)
    S0[lesion] = 0.85; T2s[lesion] = 35.0; lbl[lesion] = 4

    return S0, T2s, lbl


# ─────────────── centred FFT helpers (same convention as pipeline) ────────────

def fft2c_np(x: np.ndarray) -> np.ndarray:
    """Centred FFT2 (fastMRI convention). x: complex [...,H,W]."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)),
                    axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def ifft2c_np(x: np.ndarray) -> np.ndarray:
    """Centred IFFT2. x: complex [...,H,W]."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)),
                     axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    )


def fft2c_torch(x: torch.Tensor) -> torch.Tensor:
    """Centred FFT2. x: real or complex [*, H, W]."""
    if not x.is_complex():
        x = x.to(dtype=torch.complex64)
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)),
                       dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c_torch(x: torch.Tensor) -> torch.Tensor:
    """Centred IFFT2. x: complex [*, H, W]."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)),
                        dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )


# ─────────────── undersampling mask ──────────────────────────────────────────

def make_mask_1d(W: int, acceleration: int = 4, center_fraction: float = 0.08,
                 seed: int = 42) -> np.ndarray:
    """1-D column mask: centre block + random outer lines → [W] float32."""
    mask = np.zeros(W, dtype=np.float32)
    n_center = round(W * center_fraction)
    c0 = (W - n_center) // 2
    mask[c0: c0 + n_center] = 1.0

    outer = np.concatenate([np.arange(c0), np.arange(c0 + n_center, W)])
    n_outer = max(1, round(len(outer) / acceleration))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(outer, size=min(n_outer, len(outer)), replace=False)
    mask[chosen] = 1.0
    return mask


# ─────────────── noise ───────────────────────────────────────────────────────

def add_kspace_noise(kspace: np.ndarray, snr_db: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Add complex Gaussian noise to k-space at the requested SNR (dB)."""
    sig_power = np.mean(np.abs(kspace) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    std = np.sqrt(noise_power / 2)   # per real/imag channel
    noise = std * (rng.standard_normal(kspace.shape)
                   + 1j * rng.standard_normal(kspace.shape))
    return (kspace + noise).astype(np.complex64)


# ─────────────── synthesise phantom data ─────────────────────────────────────

def synthesise(S0: np.ndarray, T2s: np.ndarray, TEs: list[float],
               accel: int = 4, cf: float = 0.08,
               snr_db: float = 30.0, mask_seed: int = 42,
               noise_seed: int = 0):
    """
    Build everything needed for training + evaluation from the phantom.

    Returns a dict with:
      echoes_full  [n_echoes, H, W] float32  — ground-truth magnitude echoes
      kspace_full  [n_echoes, H, W] complex64
      kspace_under [n_echoes, H, W] complex64
      mask_1d      [W] float32
      mask_2d      [H, W] float32
      zf_mag       [n_echoes, H, W] float32  — |ifft2c(kspace_under)|
    """
    H, W = S0.shape
    rng = np.random.default_rng(noise_seed)

    echoes_full  = np.stack([S0 * np.exp(-te / T2s) for te in TEs], axis=0).astype(np.float32)
    kspace_full  = np.stack([fft2c_np(e.astype(np.complex64)) for e in echoes_full], axis=0)

    mask_1d = make_mask_1d(W, accel, cf, seed=mask_seed)
    mask_2d = np.broadcast_to(mask_1d[None, :], (H, W)).copy().astype(np.float32)

    kspace_under = np.stack(
        [add_kspace_noise(k * mask_2d, snr_db, rng) for k in kspace_full], axis=0
    )
    zf_mag = np.abs(np.stack([ifft2c_np(k) for k in kspace_under], axis=0)).astype(np.float32)

    return dict(
        echoes_full=echoes_full,
        kspace_full=kspace_full,
        kspace_under=kspace_under,
        mask_1d=mask_1d,
        mask_2d=mask_2d,
        zf_mag=zf_mag,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — BOTTLENECK NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class BottleneckNet(nn.Module):
    """
    Small CNN: [B, n_echoes, H, W] ZF-magnitude → S0 [B,1,H,W], T2* [B,1,H,W].

    S0  constrained positive via softplus.
    T2* constrained to [T2_MIN, T2_MAX] ms via sigmoid scaling.
    """

    def __init__(self, n_echoes: int = 5, hidden: int = 48,
                 T2_min: float = 5.0, T2_max: float = 150.0) -> None:
        super().__init__()
        self.T2_min = T2_min
        self.T2_max = T2_max

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1),
                nn.GroupNorm(min(8, cout), cout),
                nn.GELU(),
            )

        self.enc = nn.Sequential(
            block(n_echoes, hidden),
            block(hidden, hidden),
            block(hidden, hidden * 2),
            block(hidden * 2, hidden),
            block(hidden, hidden),
        )
        self.s0_head = nn.Conv2d(hidden, 1, 1)
        self.t2_head = nn.Conv2d(hidden, 1, 1)

        # Initialise output biases so default predictions are near mean values
        with torch.no_grad():
            # softplus(b) ≈ 0.6  →  b ≈ log(exp(0.6)-1) ≈ 0.37
            nn.init.constant_(self.s0_head.bias, 0.37)
            # sigmoid(b)*(T2_max-T2_min)+T2_min = 40 → sigmoid(b)=0.241 → b≈-1.15
            nn.init.constant_(self.t2_head.bias, -1.15)

    def forward(self, x: torch.Tensor):
        feat = self.enc(x)
        s0 = F.softplus(self.s0_head(feat))
        t2 = self.T2_min + (self.T2_max - self.T2_min) * torch.sigmoid(self.t2_head(feat))
        return s0, t2   # each [B, 1, H, W]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────── losses ───────────────────────────────────────────────────────

def kspace_consistency_loss(s0: torch.Tensor, t2: torch.Tensor,
                             k_under_tc: torch.Tensor,
                             mask_tc: torch.Tensor,
                             TEs_ms: list[float]) -> torch.Tensor:
    """
    Physics-only training loss: measured-line residual across all echoes.

    s0, t2 : [1, 1, H, W] — predicted parameter maps
    k_under_tc : [n_echoes, H, W] complex
    mask_tc    : [H, W] float
    """
    s0_ = s0[0, 0]   # [H, W]
    t2_ = t2[0, 0]   # [H, W]

    total = torch.zeros(1, device=s0.device)
    norm = torch.zeros(1, device=s0.device)

    for e, te in enumerate(TEs_ms):
        echo_hat = s0_ * torch.exp(-te / t2_)           # [H, W] real
        k_hat = fft2c_torch(echo_hat)                    # [H, W] complex
        k_meas = k_under_tc[e]                           # [H, W] complex

        res = mask_tc * (k_hat - k_meas)
        total = total + (res.real ** 2 + res.imag ** 2).sum()
        norm = norm + (mask_tc * k_meas.abs() ** 2).sum()

    return total / (norm + 1e-8)


def supervised_loss(s0: torch.Tensor, t2: torch.Tensor,
                    s0_gt: torch.Tensor, t2s_gt: torch.Tensor) -> torch.Tensor:
    """Direct MSE on S0 and T2* maps."""
    return F.mse_loss(s0[0, 0], s0_gt) + F.mse_loss(t2[0, 0] / 75.0, t2s_gt / 75.0)


# ─────────────── training loop ────────────────────────────────────────────────

def train(model: BottleneckNet,
          zf_input: torch.Tensor,          # [1, n_echoes, H, W]
          k_under_tc: torch.Tensor,        # [n_echoes, H, W] complex
          mask_tc: torch.Tensor,           # [H, W]
          TEs_ms: list[float],
          s0_gt: torch.Tensor | None,      # [H, W] — None for physics-only
          t2s_gt: torch.Tensor | None,     # [H, W]
          n_epochs: int = 500,
          lr: float = 1e-3,
          device: torch.device = torch.device("cpu"),
          verbose: bool = False,
          label: str = "") -> BottleneckNet:

    model = model.to(device)
    zf_input   = zf_input.to(device)
    k_under_tc = k_under_tc.to(device)
    mask_tc    = mask_tc.to(device)
    supervised = s0_gt is not None

    if supervised:
        s0_gt  = s0_gt.to(device)
        t2s_gt = t2s_gt.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    log_interval = max(1, n_epochs // 5)

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()

        s0_hat, t2_hat = model(zf_input)

        if supervised:
            loss = supervised_loss(s0_hat, t2_hat, s0_gt, t2s_gt)
        else:
            loss = kspace_consistency_loss(s0_hat, t2_hat, k_under_tc, mask_tc, TEs_ms)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if verbose and epoch % log_interval == 0:
            print(f"  [{label}] epoch {epoch:4d}/{n_epochs}  loss={loss.item():.5f}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — BASELINES + EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def analytical_fit(echoes: np.ndarray, TEs: list[float],
                   eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel log-linear fit: log(S) = log(S0) − TE/T2*.
    Matches the logic in SimplifiedBlochLoss.predict_t2_star_map.

    echoes: [n_echoes, H, W] float  (can be ZF-magnitude or fully-sampled)
    Returns S0_fit [H,W], T2s_fit [H,W] (ms)
    """
    TEs_a = np.array(TEs, dtype=np.float64)   # [n]
    n = len(TEs_a)

    log_e = np.log(np.clip(echoes, eps, None).astype(np.float64))  # [n, H, W]

    sum_x  = TEs_a.sum()                              # scalar
    sum_x2 = (TEs_a ** 2).sum()                       # scalar
    sum_y  = log_e.sum(axis=0)                        # [H, W]
    sum_xy = (TEs_a[:, None, None] * log_e).sum(axis=0)   # [H, W]

    det = n * sum_x2 - sum_x ** 2                     # scalar
    slope     = (n * sum_xy - sum_x * sum_y) / (det + eps)   # [H, W]
    intercept = (sum_y - sum_x * slope) / n                   # [H, W]

    T2s_fit = np.clip(-1.0 / (slope - eps), 1.0, 300.0).astype(np.float32)
    S0_fit  = np.exp(intercept).astype(np.float32)
    return S0_fit, T2s_fit


def predict_from_model(model: BottleneckNet,
                       zf_input: torch.Tensor,
                       device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (S0_pred [H,W], T2s_pred [H,W]) in numpy float32."""
    model.eval()
    with torch.no_grad():
        s0, t2 = model(zf_input.to(device))
    return s0[0, 0].cpu().numpy(), t2[0, 0].cpu().numpy()


def compute_metrics(pred_T2: np.ndarray, pred_S0: np.ndarray,
                    true_T2: np.ndarray, true_S0: np.ndarray,
                    fg_mask: np.ndarray) -> dict:
    """
    Metrics evaluated only on foreground (non-background) pixels.

    Returns dict with keys: t2_med, t2_p95, t2_mae, s0_med_pct, s0_mae_pct
    """
    t2_err = np.abs(pred_T2[fg_mask] - true_T2[fg_mask])
    s0_err = np.abs(pred_S0[fg_mask] - true_S0[fg_mask]) / (true_S0[fg_mask] + 1e-8) * 100.0
    return dict(
        t2_med=float(np.median(t2_err)),
        t2_p95=float(np.percentile(t2_err, 95)),
        t2_mae=float(t2_err.mean()),
        s0_med_pct=float(np.median(s0_err)),
        s0_mae_pct=float(s0_err.mean()),
    )


# ─────────────── figure helpers ───────────────────────────────────────────────

_VMIN_T2, _VMAX_T2 = 0.0, 80.0
_VMIN_S0, _VMAX_S0 = 0.0, 1.2

def save_figure(tag: str,
                true_T2: np.ndarray, pred_T2: np.ndarray,
                true_S0: np.ndarray, pred_S0: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(tag, fontsize=13)

    def im(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
        i = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(i, ax=ax, fraction=0.046, pad=0.04)

    im(axes[0, 0], true_T2, "GT T2* (ms)",       _VMIN_T2, _VMAX_T2)
    im(axes[0, 1], pred_T2, "Pred T2* (ms)",      _VMIN_T2, _VMAX_T2)
    im(axes[0, 2], np.abs(pred_T2 - true_T2),
       "│Error│ T2* (ms)", 0, 20, cmap="hot")

    im(axes[1, 0], true_S0, "GT S0",              _VMIN_S0, _VMAX_S0)
    im(axes[1, 1], pred_S0, "Pred S0",            _VMIN_S0, _VMAX_S0)
    im(axes[1, 2], np.abs(pred_S0 - true_S0),
       "│Error│ S0", 0, 0.3, cmap="hot")

    fig.tight_layout()
    out = RESULTS_DIR / f"{tag.replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → saved {out.name}")


def save_crossseed_figure(t2_stack: np.ndarray, true_T2: np.ndarray) -> None:
    """t2_stack: [N_SEEDS, H, W]"""
    std_map = t2_stack.std(axis=0)
    mean_map = t2_stack.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Cross-seed T2* variability (physics-only)", fontsize=11)

    def im(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
        i = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=9); ax.axis("off")
        plt.colorbar(i, ax=ax, fraction=0.046, pad=0.04)

    im(axes[0], true_T2,  "GT T2* (ms)",          _VMIN_T2, _VMAX_T2)
    im(axes[1], mean_map, "Mean pred T2* (ms)",    _VMIN_T2, _VMAX_T2)
    im(axes[2], std_map,  "Seed std T2* (ms)",     0, 10, cmap="hot")

    fig.tight_layout()
    out = RESULTS_DIR / "crossseed_T2.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → saved {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GATE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_gate(n_epochs: int = 500, snr_db: float = 30.0) -> None:
    # ── device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # ── PART 1: Phantom & data ────────────────────────────────────────────────
    print("\n══ PART 1 — Phantom & forward model ══")
    S0_gt, T2s_gt, labels = make_phantom(H, W)
    fg_mask = labels > 0                          # non-background pixels

    data = synthesise(S0_gt, T2s_gt, TEs_MS,
                      accel=ACCEL, cf=CF,
                      snr_db=snr_db,
                      mask_seed=42, noise_seed=7)

    print(f"  Phantom : {H}×{W}, {N_ECHOES} echoes @ TEs={TEs_MS} ms")
    print(f"  Mask    : {int(data['mask_1d'].sum())}/{W} columns sampled "
          f"(~{int(data['mask_1d'].sum())/W*100:.0f}% = {W/int(data['mask_1d'].sum()):.1f}× accel)")
    print(f"  SNR     : {snr_db} dB")
    print(f"  GT T2*  : range {T2s_gt[fg_mask].min():.0f}–{T2s_gt[fg_mask].max():.0f} ms "
          f"(fg pixels {fg_mask.sum()})")

    # Torch tensors for training
    zf_input_np = data["zf_mag"]                          # [n_echoes, H, W] float32
    zf_input_tc = torch.from_numpy(zf_input_np).unsqueeze(0)   # [1, n_echoes, H, W]

    k_under_tc = torch.from_numpy(data["kspace_under"]).to(torch.complex64)  # [n_echoes, H, W]
    mask_tc    = torch.from_numpy(data["mask_2d"])                            # [H, W]
    S0_gt_tc   = torch.from_numpy(S0_gt)                                      # [H, W]
    T2s_gt_tc  = torch.from_numpy(T2s_gt)                                     # [H, W]

    results: dict[str, dict] = {}

    # ── METHOD 1: Analytical — fully-sampled ─────────────────────────────────
    print("\n══ METHOD 1 — Analytical fit (fully-sampled) ══")
    S0_ana_full, T2s_ana_full = analytical_fit(data["echoes_full"], TEs_MS)
    m1 = compute_metrics(T2s_ana_full, S0_ana_full, T2s_gt, S0_gt, fg_mask)
    results["1_analytical_full"] = m1
    save_figure("1 Analytical (fully-sampled)", T2s_gt, T2s_ana_full, S0_gt, S0_ana_full)
    print(f"  T2* median error : {m1['t2_med']:.2f} ms  (p95={m1['t2_p95']:.2f})")
    print(f"  S0  median error : {m1['s0_med_pct']:.2f} %")

    # ── METHOD 2: Analytical — zero-filled ───────────────────────────────────
    print("\n══ METHOD 2 — Analytical fit (zero-filled) ══")
    S0_ana_zf, T2s_ana_zf = analytical_fit(data["zf_mag"], TEs_MS)
    m2 = compute_metrics(T2s_ana_zf, S0_ana_zf, T2s_gt, S0_gt, fg_mask)
    results["2_analytical_zf"] = m2
    save_figure("2 Analytical (zero-filled)", T2s_gt, T2s_ana_zf, S0_gt, S0_ana_zf)
    print(f"  T2* median error : {m2['t2_med']:.2f} ms  (p95={m2['t2_p95']:.2f})")
    print(f"  S0  median error : {m2['s0_med_pct']:.2f} %")

    # ── METHOD 3: Physics-only bottleneck (N_SEEDS runs) ─────────────────────
    print(f"\n══ METHOD 3 — Physics-only bottleneck ({N_SEEDS} seeds) ══")
    print(f"  Epochs: {n_epochs}, model params: {BottleneckNet(N_ECHOES, HIDDEN).n_params():,}")

    seed_T2_maps: list[np.ndarray] = []
    seed_S0_maps: list[np.ndarray] = []
    seed_metrics: list[dict]       = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = BottleneckNet(N_ECHOES, HIDDEN, T2_MIN, T2_MAX)
        model = train(
            model, zf_input_tc, k_under_tc, mask_tc, TEs_MS,
            s0_gt=None, t2s_gt=None,
            n_epochs=n_epochs, lr=1e-3,
            device=device, verbose=True, label=f"phys seed={seed}"
        )

        S0_pred, T2s_pred = predict_from_model(model, zf_input_tc, device)
        seed_T2_maps.append(T2s_pred)
        seed_S0_maps.append(S0_pred)
        seed_metrics.append(compute_metrics(T2s_pred, S0_pred, T2s_gt, S0_gt, fg_mask))
        print(f"  seed={seed}  T2*_med_err={seed_metrics[-1]['t2_med']:.2f} ms  "
              f"S0_med_err={seed_metrics[-1]['s0_med_pct']:.2f}%")
        save_figure(f"3 Physics-only seed={seed}",
                    T2s_gt, T2s_pred, S0_gt, S0_pred)

    # Best seed for the main table
    best_seed = int(np.argmin([m["t2_med"] for m in seed_metrics]))
    m3 = seed_metrics[best_seed]
    results["3_physics_only"] = m3

    # Cross-seed stats
    t2_stack = np.stack(seed_T2_maps, axis=0)                    # [N_SEEDS, H, W]
    cross_seed_std_fg = t2_stack[:, fg_mask].std(axis=0)         # [N_SEEDS, fg_pixels]
    mean_cross_std = float(cross_seed_std_fg.mean())
    max_cross_std  = float(cross_seed_std_fg.max())
    save_crossseed_figure(t2_stack, T2s_gt)

    # ── METHOD 4: Supervised ──────────────────────────────────────────────────
    print(f"\n══ METHOD 4 — Supervised bottleneck ══")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model_sup = BottleneckNet(N_ECHOES, HIDDEN, T2_MIN, T2_MAX)
    model_sup = train(
        model_sup, zf_input_tc, k_under_tc, mask_tc, TEs_MS,
        s0_gt=S0_gt_tc, t2s_gt=T2s_gt_tc,
        n_epochs=n_epochs, lr=1e-3,
        device=device, verbose=True, label="supervised"
    )
    S0_sup, T2s_sup = predict_from_model(model_sup, zf_input_tc, device)
    m4 = compute_metrics(T2s_sup, S0_sup, T2s_gt, S0_gt, fg_mask)
    results["4_supervised"] = m4
    save_figure("4 Supervised", T2s_gt, T2s_sup, S0_gt, S0_sup)
    print(f"  T2* median error : {m4['t2_med']:.2f} ms  (p95={m4['t2_p95']:.2f})")
    print(f"  S0  median error : {m4['s0_med_pct']:.2f} %")

    # ═══════════════════════════════════════════════════════════════════════════
    # VERDICT TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 72)
    print("IDENTIFIABILITY GATE — VERDICT TABLE")
    print("═" * 72)
    header = f"{'Method':<38}  {'T2* med':>8}  {'T2* p95':>8}  {'S0 med%':>8}"
    print(header)
    print("─" * 72)
    rows = [
        ("1. Analytical (fully-sampled) [UB]", results["1_analytical_full"]),
        ("2. Analytical (zero-filled)   [NB]", results["2_analytical_zf"]),
        (f"3. Physics-only (best/{N_SEEDS} seeds) [TEST]", results["3_physics_only"]),
        ("4. Supervised                 [SC]",  results["4_supervised"]),
    ]
    for label, m in rows:
        print(f"  {label:<36}  {m['t2_med']:>7.2f}  {m['t2_p95']:>7.2f}  {m['s0_med_pct']:>7.2f}")
    print("─" * 72)
    print(f"\n  Cross-seed T2* std (mean over fg): {mean_cross_std:.3f} ms")
    print(f"  Cross-seed T2* std (max  over fg): {max_cross_std:.3f} ms")

    per_seed_str = "  ".join(
        f"seed{i}={seed_metrics[i]['t2_med']:.2f}" for i in range(N_SEEDS)
    )
    print(f"\n  Per-seed T2* median errors: {per_seed_str}")

    # ── PASS / FAIL decision ──────────────────────────────────────────────────
    beats_zf     = m3["t2_med"] < m2["t2_med"]
    below_5ms    = m3["t2_med"] < 5.0
    low_variance = mean_cross_std < 2.0
    sup_passes   = m4["t2_med"] < m2["t2_med"]

    print("\n  Criteria (physics-only, best seed):")
    print(f"    T2* median < 5 ms             : {'PASS' if below_5ms   else 'FAIL'}  ({m3['t2_med']:.2f} ms)")
    print(f"    Beats ZF baseline (method 2)   : {'PASS' if beats_zf    else 'FAIL'}  ({m3['t2_med']:.2f} < {m2['t2_med']:.2f})")
    print(f"    Cross-seed std < 2 ms          : {'PASS' if low_variance else 'FAIL'}  ({mean_cross_std:.3f} ms)")
    print(f"    Supervised passes (arch check) : {'PASS' if sup_passes   else 'FAIL'}  ({m4['t2_med']:.2f} ms)")

    n_pass = sum([below_5ms, beats_zf, low_variance, sup_passes])
    if n_pass == 4:
        verdict = "PASS — network IS identifiable from k-space consistency alone"
    elif not sup_passes:
        verdict = "FAIL — architecture cannot fit maps even with GT supervision"
    elif not beats_zf:
        verdict = "FAIL — physics-only does NOT beat naive ZF baseline"
    elif not low_variance:
        verdict = "FAIL — high cross-seed variance suggests non-unique solutions"
    else:
        verdict = f"PARTIAL ({n_pass}/4) — identifiable but T2* error > 5 ms threshold"

    print(f"\n  VERDICT: {verdict}")
    print("═" * 72 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Identifiability Gate Experiment")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs for bottleneck nets (default 500)")
    parser.add_argument("--snr", type=float, default=30.0,
                        help="K-space noise SNR in dB (default 30)")
    args = parser.parse_args()

    run_gate(n_epochs=args.epochs, snr_db=args.snr)


if __name__ == "__main__":
    main()
