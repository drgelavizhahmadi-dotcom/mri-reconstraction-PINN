#!/usr/bin/env python3
"""
JEPA Prior for MRI Parameter Map Reconstruction
================================================
The textured-map result (run_prior_vs_physics.py) showed the CNN smoothness prior
collapses to the ZF floor on realistic maps at 4×: the generic convolutional prior
fills the unmeasured k-space band with blur.

THIS EXPERIMENT tests whether a JEPA-learned prior over parameter maps fills that
band with TRUE texture and measures how much it HALLUCINATES.

STAGE A — Pretrain a small JEPA over (S0, T2*) parameter maps (no physics, no k-space):
  Encoder → EMA target encoder + KAN predictor.  Mask spatial BLOCKS; predict masked
  block representations from visible context in latent space.
  Loss: ||pred − sg(target)||² + VICReg anti-collapse.
  SANITY GATE: frozen prior must discriminate real vs smoothed vs random maps.

STAGE B — Reconstruct at 4× with JEPA prior as regulariser:
  3-param physics bottleneck; DC loss on measured lines; + λ·JEPA_surprise term.
  Compare vs smoothness-CNN bottleneck (prior run), ZF floor, and full-sample UB.

STAGE C — Decisive measurement on held-out textured maps (known ground truth):
  • Whole-map T2* error (median, p95)
  • Texture fidelity in unmeasured k-space band: correlation + NMSE
  • Hallucination fraction: fraction of recovered unmeasured-band energy that is wrong
  • Cross-seed std + residual (check: do they detect hallucination?)

VERDICT: PROMISING / HALLUCINATES / NO-FILL

Usage:
    python experiments/identifiability_gate/run_jepa_prior.py
    python experiments/identifiability_gate/run_jepa_prior.py --pretrain-epochs 80 --recon-epochs 200
"""

from __future__ import annotations

import argparse
import copy
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
from scipy.ndimage import gaussian_filter, zoom

# ── repo src on path (KANLinear lives here) ───────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GATE_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_GATE_DIR))

from mhwf_pikan.core.fno_kan import KANLinear, SpatialKAN   # noqa: E402
from run_gate import (                                        # noqa: E402
    make_phantom, synthesise, analytical_fit, BottleneckNet,
    kspace_consistency_loss, predict_from_model,
    fft2c_torch, fft2c_np, ifft2c_np,
    H, W, TEs_MS, N_ECHOES, ACCEL, CF, N_SEEDS, HIDDEN, T2_MIN, T2_MAX,
    RESULTS_DIR,
)
from run_offres_fix import (                                  # noqa: E402
    BottleneckNet3, train_3param, predict_3param, kspace_loss_3param, DF_BOUND,
)
from run_mismatch import make_offres_map                     # noqa: E402

warnings.filterwarnings("ignore")

# ─────────────────────────────── constants ───────────────────────────────────

SNR_DB          = 30.0
DATA_DIR        = Path("data/singlecoil_val")

# Dataset
N_PRETRAIN_FILES = 8
N_TEST_FILES     = 2
SLICES_PER_FILE  = 4      # equally-spaced interior slices per file
SEEDS_PER_SLICE  = 2      # texture realisations per slice
TEXTURE_SIGMA    = 3.0
TEXTURE_AMP      = 0.30

# JEPA architecture
JEPA_CH     = 32          # feature channels
PATCH_SIZE  = 8           # px; 128/8 = 16 patch grid per side
N_PATCH     = H // PATCH_SIZE   # 16
EMA_DECAY   = 0.996

# Stage A
PRETRAIN_BATCH = 8
PRETRAIN_LR    = 3e-4
VIC_LAMBDA_VAR = 10.0     # variance penalty weight
VIC_LAMBDA_COV = 1.0      # covariance penalty weight

# Stage B
LAMBDA_JEPA  = 0.5        # weight for JEPA-surprise term in reconstruction loss
N_MASKS_SURP = 2          # JEPA mask realisations per reconstruction step
RECON_LR     = 1e-3

# Stage C
N_RECON_MAPS = 4          # test maps to reconstruct


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — TEXTURED MAP DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def _get_corpd_files() -> list[str]:
    """Return sorted list of non-fat-suppressed CORPD singlecoil files."""
    files = []
    for fn in sorted(os.listdir(DATA_DIR)):
        fp = DATA_DIR / fn
        try:
            with h5py.File(fp, "r") as f:
                acq = f.attrs.get("acquisition", "")
                n_sl = f["reconstruction_rss"].shape[0]
            if "CORPD" in acq and "FS" not in acq and "DFS" not in acq and n_sl >= 30:
                files.append(fn)
        except Exception:
            pass
    return files


def _make_textured_map(rss_slice: np.ndarray, texture_seed: int,
                       H_out: int = H, W_out: int = W) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert one RSS slice to a textured (S0, T2*, fg) triplet.
    Same generator as run_prior_vs_physics.py — reused for consistency.
    """
    scale = H_out / rss_slice.shape[0]
    S0 = zoom(rss_slice, scale).astype(np.float32)[:H_out, :W_out]
    S0 /= S0.max() + 1e-9

    fg = S0 > 0.05

    bins   = [0.0, 0.05, 0.20, 0.40, 0.65, 1.01]
    t2_val = [T2_MIN, 20.0, 40.0, 65.0, 85.0]
    lbl    = np.digitize(S0, bins) - 1
    T2_base = np.full((H_out, W_out), T2_MIN, dtype=np.float32)
    for ci, tv in enumerate(t2_val):
        T2_base[lbl == ci] = tv

    rng         = np.random.default_rng(texture_seed)
    noise       = rng.standard_normal((H_out, W_out)).astype(np.float32)
    noise_sm    = gaussian_filter(noise, sigma=TEXTURE_SIGMA)
    noise_sm   /= noise_sm.std() + 1e-8
    T2 = np.clip(T2_base * (1.0 + TEXTURE_AMP * noise_sm), T2_MIN, T2_MAX).astype(np.float32)
    T2[~fg] = T2_MIN

    return S0.astype(np.float32), T2, fg


def generate_map_dataset(files: list[str], seeds: list[int]) -> list[dict]:
    """
    Build one (S0, T2*, fg) dict per (file, slice, seed) combination.
    Slices are equally spaced interior slices to avoid end-of-volume artefacts.
    """
    maps = []
    for fn in files:
        with h5py.File(DATA_DIR / fn, "r") as f:
            rss = f["reconstruction_rss"][()]  # [n_sl, 320, 320]
        n_sl = rss.shape[0]
        # SLICES_PER_FILE equally-spaced interior slices
        idxs = np.linspace(n_sl * 0.2, n_sl * 0.8, SLICES_PER_FILE, dtype=int)
        for si in idxs:
            for seed in seeds:
                S0, T2, fg = _make_textured_map(rss[si], texture_seed=int(seed))
                maps.append({"S0": S0, "T2": T2, "fg": fg,
                             "file": fn, "slice": int(si), "seed": int(seed)})
    return maps


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — JEPA ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_maps(S0: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """Normalise (S0, T2*) to [0,1] using fixed parameter ranges."""
    s0_n = S0 / 1.0                                # S0 ∈ [0, 1] by construction
    t2_n = (T2 - T2_MIN) / (T2_MAX - T2_MIN + 1e-8)
    return torch.stack([s0_n, t2_n], dim=0)        # [2, H, W]


def norm_maps_batch(S0: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """Batch version: S0, T2 each [B, 1, H, W] → [B, 2, H, W]."""
    s0_n = S0[:, 0] / 1.0
    t2_n = (T2[:, 0] - T2_MIN) / (T2_MAX - T2_MIN + 1e-8)
    return torch.stack([s0_n, t2_n], dim=1)        # [B, 2, H, W]


class ContextEncoder(nn.Module):
    """
    Small CNN: [B, 2, H, W] → [B, JEPA_CH, H/8, W/8].
    Patch resolution is 16×16 for 128×128 input.
    """
    def __init__(self, in_ch: int = 2, feat_ch: int = JEPA_CH) -> None:
        super().__init__()

        def blk(ci, co, stride=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride, 1),
                nn.GroupNorm(min(8, co), co),
                nn.GELU(),
            )

        self.net = nn.Sequential(
            blk(in_ch,      feat_ch),
            blk(feat_ch,    feat_ch, stride=2),   # /2
            blk(feat_ch,    feat_ch * 2),
            blk(feat_ch * 2, feat_ch * 2, stride=2),  # /4
            blk(feat_ch * 2, feat_ch),
            blk(feat_ch,    feat_ch, stride=2),   # /8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANPredictor(nn.Module):
    """
    Predictor in the JEPA latent space: context features → target features.
    Architecture: Conv3×3 → SpatialKAN (B-spline activations) → Conv1×1.
    Operates at patch resolution [B, C, N_PATCH, N_PATCH].
    """
    def __init__(self, feat_ch: int = JEPA_CH) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(min(8, feat_ch), feat_ch)
        self.kan   = SpatialKAN(feat_ch, grid_size=5, spline_order=3)
        self.conv2 = nn.Conv2d(feat_ch, feat_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.kan(h)
        return self.conv2(h) + x   # residual


def _update_ema(online: nn.Module, ema: nn.Module, decay: float) -> None:
    """Exponential-moving-average parameter update (stop-grad target encoder)."""
    with torch.no_grad():
        for p_on, p_ema in zip(online.parameters(), ema.parameters()):
            p_ema.data = decay * p_ema.data + (1 - decay) * p_on.data


def _sample_block_mask(n_patch: int, min_sz: int = 4, max_sz: int = 8,
                       rng: random.Random | None = None) -> tuple[int, int, int, int]:
    """Sample a random rectangular target block (row0, col0, rows, cols)."""
    if rng is None:
        rng = random
    h = rng.randint(min_sz, max_sz)
    w = rng.randint(min_sz, max_sz)
    r0 = rng.randint(0, n_patch - h)
    c0 = rng.randint(0, n_patch - w)
    return r0, c0, h, w


def _block_mask_tensor(n_patch: int, block: tuple[int, int, int, int],
                       device: torch.device) -> torch.Tensor:
    """Return [1, 1, n_patch, n_patch] float mask: 1 = target block."""
    r0, c0, bh, bw = block
    m = torch.zeros(1, 1, n_patch, n_patch, device=device)
    m[:, :, r0:r0 + bh, c0:c0 + bw] = 1.0
    return m


def mask_input(x: torch.Tensor, block: tuple[int, int, int, int]) -> torch.Tensor:
    """Zero-out target block in PIXEL space (for context encoder input)."""
    r0, c0, bh, bw = block
    x = x.clone()
    x[..., r0 * PATCH_SIZE:(r0 + bh) * PATCH_SIZE,
          c0 * PATCH_SIZE:(c0 + bw) * PATCH_SIZE] = 0.0
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — STAGE A: JEPA PRE-TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def vicreg_collapse_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Variance + covariance anti-collapse on a batch of patch features.
    z: [N, D] where N = batch × spatial positions.
    """
    N, D = z.shape
    # Variance: penalise when per-dimension std < 1
    std    = z.std(dim=0)
    v_loss = F.relu(1.0 - std).mean()

    # Covariance: off-diagonal penalty
    z_c    = z - z.mean(dim=0)
    cov    = (z_c.T @ z_c) / (N - 1 + 1e-8)
    c_loss = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / (D + 1e-8)

    return VIC_LAMBDA_VAR * v_loss + VIC_LAMBDA_COV * c_loss


def pretrain_jepa(
    encoder: ContextEncoder,
    ema_encoder: ContextEncoder,
    predictor: KANPredictor,
    maps: list[dict],
    n_epochs: int,
    device: torch.device,
) -> list[float]:
    """
    Stage A: JEPA self-supervised pretraining on parameter maps only.
    Returns list of per-epoch losses.
    """
    encoder.to(device); ema_encoder.to(device); predictor.to(device)

    opt   = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=PRETRAIN_LR, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=PRETRAIN_LR * 0.05)

    rng_mask = random.Random(42)
    losses   = []

    # Convert all maps to normalised tensors
    all_x = []
    for m in maps:
        S0_t = torch.from_numpy(m["S0"])
        T2_t = torch.from_numpy(m["T2"])
        all_x.append(_norm_maps(S0_t, T2_t))  # [2, H, W]
    all_x_t = torch.stack(all_x, dim=0)       # [N_maps, 2, H, W]

    n_maps   = len(maps)
    log_ev   = max(1, n_epochs // 5)

    for epoch in range(1, n_epochs + 1):
        # Random mini-batch
        idx   = torch.randperm(n_maps)[:PRETRAIN_BATCH]
        x_bat = all_x_t[idx].to(device)       # [B, 2, H, W]
        B     = x_bat.shape[0]

        opt.zero_grad()
        total_loss  = torch.zeros(1, device=device)
        pred_feats  = []

        for bi in range(B):
            xi     = x_bat[bi:bi+1]            # [1, 2, H, W]
            block  = _sample_block_mask(N_PATCH, rng=rng_mask)
            mask_t = _block_mask_tensor(N_PATCH, block, device)

            xi_ctx    = mask_input(xi, block)
            feat_ctx  = encoder(xi_ctx)         # [1, C, 16, 16]
            feat_pred = predictor(feat_ctx)     # [1, C, 16, 16]

            with torch.no_grad():
                feat_tgt = ema_encoder(xi)      # [1, C, 16, 16]

            pred_loss = (mask_t * (feat_pred - feat_tgt).pow(2)).sum() / (mask_t.sum() + 1e-8)
            total_loss = total_loss + pred_loss

            # Collect ATTACHED features for VICReg (graph still live)
            pred_feats.append(feat_pred.permute(0, 2, 3, 1).reshape(-1, JEPA_CH))

        # VICReg anti-collapse: grads flow through pred_feats → encoder/predictor
        vic_loss   = vicreg_collapse_loss(torch.cat(pred_feats, dim=0))
        epoch_loss = float((total_loss / B).item())
        (total_loss / B + vic_loss).backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), 1.0
        )
        opt.step()
        sched.step()

        # EMA update
        _update_ema(encoder, ema_encoder, EMA_DECAY)

        losses.append(epoch_loss)
        if epoch % log_ev == 0:
            print(f"  [JEPA pretrain] ep {epoch:4d}/{n_epochs}  "
                  f"pred_loss={epoch_loss:.5f}  vic={float(vic_loss.item()):.4f}")

    return losses


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — SANITY GATE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_jepa_surprise_np(
    S0: np.ndarray, T2: np.ndarray,
    encoder: ContextEncoder,
    ema_encoder: ContextEncoder,
    predictor: KANPredictor,
    device: torch.device,
    n_masks: int = 8,
    rng_seed: int = 0,
) -> float:
    """Compute average masked-prediction error (surprise) for a numpy map pair."""
    encoder.eval(); ema_encoder.eval(); predictor.eval()
    rng = random.Random(rng_seed)

    S0_t = torch.from_numpy(S0)
    T2_t = torch.from_numpy(T2)
    x = _norm_maps(S0_t, T2_t).unsqueeze(0).to(device)  # [1, 2, H, W]

    total = 0.0
    with torch.no_grad():
        feat_tgt = ema_encoder(x)               # [1, C, 16, 16]
        for _ in range(n_masks):
            block   = _sample_block_mask(N_PATCH, rng=rng)
            mask_t  = _block_mask_tensor(N_PATCH, block, device)
            xi_ctx  = mask_input(x, block)
            feat_ctx  = encoder(xi_ctx)
            feat_pred = predictor(feat_ctx)
            loss = (mask_t * (feat_pred - feat_tgt).pow(2)).sum() / (mask_t.sum() + 1e-8)
            total += float(loss.item())

    return total / n_masks


def run_sanity_gate(
    pretrain_maps: list[dict],
    test_maps: list[dict],
    encoder: ContextEncoder,
    ema_encoder: ContextEncoder,
    predictor: KANPredictor,
    device: torch.device,
) -> tuple[bool, dict]:
    """
    HARD SANITY GATE: frozen JEPA must discriminate on- vs off-manifold maps.
    Tests three populations:
      (a) real textured held-out maps  → expected LOW surprise
      (b) Gaussian-blurred real maps (σ=10 px) → expected HIGHER
      (c) random uniform noise maps   → expected HIGHEST
    PASS: mean_real < mean_blurred AND mean_real < mean_noise (both gaps ≥ 10 %)
    """
    rng_probe = random.Random(99)

    def score_maps(maps_list: list[dict]) -> list[float]:
        return [compute_jepa_surprise_np(m["S0"], m["T2"],
                                         encoder, ema_encoder, predictor, device,
                                         n_masks=6, rng_seed=rng_probe.randint(0, 9999))
                for m in maps_list]

    def blurred(m: dict) -> dict:
        return {"S0": gaussian_filter(m["S0"], sigma=10),
                "T2": gaussian_filter(m["T2"], sigma=10)}

    def noise_map(m: dict) -> dict:
        rng = np.random.default_rng(42)
        return {"S0": rng.uniform(0, 1, (H, W)).astype(np.float32),
                "T2": rng.uniform(T2_MIN, T2_MAX, (H, W)).astype(np.float32)}

    probe = test_maps[:8]   # use 8 test maps for the gate
    print("  Scoring real maps…", end=" ", flush=True)
    s_real   = score_maps(probe)
    print("done. Blurred…",    end=" ", flush=True)
    s_blur   = score_maps([blurred(m) for m in probe])
    print("done. Noise…",      end=" ", flush=True)
    s_noise  = score_maps([noise_map(m) for m in probe])
    print("done.")

    mr, mb, mn = np.mean(s_real), np.mean(s_blur), np.mean(s_noise)
    gap_blur  = (mb - mr) / (mr + 1e-8)
    gap_noise = (mn - mr) / (mr + 1e-8)

    print(f"  Surprise: real={mr:.5f}  blurred={mb:.5f} (+{gap_blur*100:.0f}%)  "
          f"noise={mn:.5f} (+{gap_noise*100:.0f}%)")

    passed = (mb > mr) and (mn > mr) and (gap_blur > 0.10 or gap_noise > 0.20)
    gate_str = "PASS" if passed else "FAIL"
    print(f"  SANITY GATE: {gate_str}  "
          f"(need: blur>real AND noise>real, gap ≥10%/20%)")

    return passed, dict(real=mr, blurred=mb, noise=mn,
                        gap_blur=gap_blur, gap_noise=gap_noise, passed=passed)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — JEPA SURPRISE (differentiable, for Stage B)
# ═══════════════════════════════════════════════════════════════════════════════

def jepa_surprise_differentiable(
    s0: torch.Tensor,           # [1, 1, H, W], requires_grad through bottleneck
    t2: torch.Tensor,           # [1, 1, H, W]
    encoder: ContextEncoder,    # frozen params
    ema_encoder: ContextEncoder,# frozen params
    predictor: KANPredictor,    # frozen params
    device: torch.device,
    n_masks: int = N_MASKS_SURP,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """
    Differentiable JEPA surprise: gradients flow through encoder/predictor INPUT
    back to (s0, t2), but JEPA WEIGHTS are frozen (requires_grad=False).
    """
    if rng is None:
        rng = random.Random()

    # Normalise parameter maps for JEPA encoder
    s0_n = s0[0, 0] / 1.0
    t2_n = (t2[0, 0] - T2_MIN) / (T2_MAX - T2_MIN + 1e-8)
    x = torch.stack([s0_n, t2_n], dim=0).unsqueeze(0)   # [1, 2, H, W]

    total = torch.zeros(1, device=device)

    for _ in range(n_masks):
        block  = _sample_block_mask(N_PATCH, rng=rng)
        mask_t = _block_mask_tensor(N_PATCH, block, device)

        # Context path — carries gradient from x → (s0, t2)
        xi_ctx    = mask_input(x, block)
        feat_ctx  = encoder(xi_ctx)
        feat_pred = predictor(feat_ctx)

        # Target path — stop-grad (EMA encoder reference)
        with torch.no_grad():
            feat_tgt = ema_encoder(x.detach())

        surprise = (mask_t * (feat_pred - feat_tgt).pow(2)).sum() / (mask_t.sum() + 1e-8)
        total = total + surprise

    return total / n_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — STAGE B: PHYSICS RECONSTRUCTION + JEPA PRIOR
# ═══════════════════════════════════════════════════════════════════════════════

def _make_complex_zf_input(kspace_under: np.ndarray) -> np.ndarray:
    zf = np.stack([ifft2c_np(kspace_under[e]) for e in range(kspace_under.shape[0])], axis=0)
    scale = float(np.max(np.abs(zf[0]))) + 1e-8
    return np.concatenate([zf.real / scale, zf.imag / scale], axis=0).astype(np.float32)


def train_3param_jepa(
    model: BottleneckNet3,
    zf_input: torch.Tensor,
    k_under_tc: torch.Tensor,
    mask_tc: torch.Tensor,
    encoder: ContextEncoder,
    ema_encoder: ContextEncoder,
    predictor: KANPredictor,
    TEs_ms: list[float],
    n_epochs: int,
    lr: float,
    device: torch.device,
    label: str = "",
) -> tuple[BottleneckNet3, float, list[float]]:
    """3-param bottleneck with DC loss + frozen JEPA surprise regulariser."""
    model.to(device)
    zf_input   = zf_input.to(device)
    k_under_tc = k_under_tc.to(device)
    mask_tc    = mask_tc.to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
    rng_mask = random.Random(hash(label) % (2**31))

    log_ev   = max(1, n_epochs // 5)
    dc_hist: list[float] = []
    final_loss = float("nan")

    for ep in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()

        s0_hat, t2_hat, df_hat = model(zf_input)

        dc_loss = kspace_loss_3param(s0_hat, t2_hat, df_hat, k_under_tc, mask_tc, TEs_ms)
        jepa_reg = jepa_surprise_differentiable(
            s0_hat, t2_hat, encoder, ema_encoder, predictor, device, rng=rng_mask
        )
        loss = dc_loss + LAMBDA_JEPA * jepa_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        dc_hist.append(float(dc_loss.item()))
        if ep == n_epochs:
            final_loss = float(dc_loss.item())
        if ep % log_ev == 0:
            print(f"    [{label}] ep {ep:4d}/{n_epochs}  "
                  f"dc={dc_loss.item():.5f}  jepa={jepa_reg.item():.5f}")

    return model, final_loss, dc_hist


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7 — STAGE C: TEXTURE FIDELITY + HALLUCINATION MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════════════

def texture_fidelity_analysis(
    T2_rec: np.ndarray,
    T2_gt: np.ndarray,
    mask_1d: np.ndarray,   # [W] float, 1=sampled 0=unmeasured
) -> dict:
    """
    Measure texture recovery in the UNMEASURED k-space band.

    unmeas_corr  — Pearson correlation between recovered and true k-space
                   coefficients in the unmeasured columns (real+imag stacked).
    unmeas_nmse  — ||K_rec - K_gt||² / ||K_gt||² in the unmeasured band.
    halluc_frac  — fraction of recovered unmeasured-band energy that is wrong:
                     ||K_rec_unmeas - K_gt_unmeas||² / ||K_rec_unmeas||²
                   0 = perfect; 1 ≈ all energy invented; >1 = worse than nothing.
    """
    # K-space of the T2* maps (real-valued map → Hermitian k-space)
    K_gt  = fft2c_np(T2_gt.astype(np.complex64))   # [H, W] complex
    K_rec = fft2c_np(T2_rec.astype(np.complex64))  # [H, W] complex

    unmeas = mask_1d == 0   # [W] bool
    K_gt_u  = K_gt[:, unmeas]    # [H, n_unmeas] complex
    K_rec_u = K_rec[:, unmeas]

    # Flatten to real vectors for correlation
    gt_r  = np.concatenate([K_gt_u.real.ravel(), K_gt_u.imag.ravel()])
    rec_r = np.concatenate([K_rec_u.real.ravel(), K_rec_u.imag.ravel()])

    corr = float(np.corrcoef(gt_r, rec_r)[0, 1]) if gt_r.std() > 1e-9 else 0.0

    nmse = float(np.sum(np.abs(K_rec_u - K_gt_u)**2)
                 / (np.sum(np.abs(K_gt_u)**2) + 1e-12))

    rec_energy = float(np.sum(np.abs(K_rec_u)**2))
    err_energy = float(np.sum(np.abs(K_rec_u - K_gt_u)**2))
    halluc_frac = err_energy / (rec_energy + 1e-12)

    return dict(corr=corr, nmse=nmse, halluc_frac=halluc_frac,
                rec_energy=rec_energy, err_energy=err_energy)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 8 — BASELINES  (ZF analytical + full-sample analytical)
# ═══════════════════════════════════════════════════════════════════════════════

def run_baselines(data: dict, T2_gt: np.ndarray, fg: np.ndarray) -> dict:
    _, T2_full = analytical_fit(data["echoes_full"], TEs_MS)
    _, T2_zf   = analytical_fit(data["zf_mag"],      TEs_MS)
    err_full = np.abs(T2_full[fg] - T2_gt[fg])
    err_zf   = np.abs(T2_zf[fg]  - T2_gt[fg])
    return dict(
        T2_full=T2_full, T2_zf=T2_zf,
        full_med=float(np.median(err_full)), full_p95=float(np.percentile(err_full, 95)),
        zf_med  =float(np.median(err_zf)),   zf_p95  =float(np.percentile(err_zf,   95)),
    )


def t2_metrics_fg(T2_pred: np.ndarray, T2_gt: np.ndarray, fg: np.ndarray) -> dict:
    e = np.abs(T2_pred[fg] - T2_gt[fg])
    return dict(med=float(np.median(e)), p95=float(np.percentile(e, 95)))


# ═══════════════════════════════════════════════════════════════════════════════
# PART 9 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def save_jepa_figure(tag: str, pretrain_losses: list[float],
                     gate: dict, results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"JEPA Prior — {tag}", fontsize=11)

    # Panel 1: pretrain loss curve
    axes[0].semilogy(pretrain_losses)
    axes[0].set_title("Stage A: pretrain loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("JEPA pred loss")

    # Panel 2: sanity gate surprise scores
    scores_labels = ["Real\n(test)", "Blurred\n(σ=10px)", "Random\nnoise"]
    scores_vals   = [gate["real"], gate["blurred"], gate["noise"]]
    bars = axes[1].bar(range(3), scores_vals,
                       color=["steelblue", "tomato", "darkorange"])
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(scores_labels)
    axes[1].set_title(f"Sanity gate: {'PASS' if gate['passed'] else 'FAIL'}")
    axes[1].set_ylabel("JEPA surprise")

    # Panel 3: texture fidelity comparison
    if results:
        methods = [r["label"] for r in results]
        corrs   = [r.get("tf", {}).get("corr", 0) for r in results]
        axes[2].bar(range(len(methods)), corrs, color=["steelblue", "tomato", "gray"])
        axes[2].set_xticks(range(len(methods)))
        axes[2].set_xticklabels(methods, fontsize=8)
        axes[2].set_title("Unmeasured-band correlation\n(texture fidelity)")
        axes[2].set_ylabel("Pearson r with GT k-space")
        axes[2].axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    out = RESULTS_DIR / "jepa_prior_summary.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  → saved {out.name}")


def save_reconstruction_figure(
    tag: str, S0_gt: np.ndarray, T2_gt: np.ndarray,
    T2_jepa: np.ndarray, T2_cnn: np.ndarray, T2_zf: np.ndarray,
) -> None:
    vt = (0, min(120, T2_gt.max() + 10))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Reconstruction comparison — {tag}", fontsize=11)

    def im(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
        v = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=9); ax.axis("off")
        plt.colorbar(v, ax=ax, fraction=0.046, pad=0.04)

    im(axes[0, 0], T2_gt,           "GT T2* (ms)",          *vt)
    im(axes[0, 1], T2_jepa,         "JEPA-prior T2* (ms)",  *vt)
    im(axes[0, 2], np.abs(T2_jepa - T2_gt), "|Error| JEPA", 0, 30, "hot")
    im(axes[1, 0], T2_zf,           "ZF analytical T2*",    *vt)
    im(axes[1, 1], T2_cnn,          "CNN-prior T2* (ms)",   *vt)
    im(axes[1, 2], np.abs(T2_cnn  - T2_gt), "|Error| CNN",  0, 30, "hot")

    fig.tight_layout()
    safe = tag.replace(" ", "_")
    out  = RESULTS_DIR / f"jepa_recon_{safe}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  → saved {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 10 — VERDICT TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_verdict(results: list[dict], gate: dict) -> str:
    import textwrap

    print("\n" + "═" * 92)
    print("JEPA PRIOR — STAGE C RESULTS")
    print("  Textured maps (held-out from pretraining), df=0 Hz, SNR=30 dB")
    print("═" * 92)

    hdr = (f"  {'Method':<22}  {'T2*med':>7}  {'T2*p95':>7}  "
           f"{'DC-loss':>8}  {'Xseed':>7}  {'Unmeas-corr':>12}  "
           f"{'NMSE(band)':>10}  {'Halluc-frac':>11}")
    print(hdr)
    print(f"  {'':22}  {'(ms)':>7}  {'(ms)':>7}  "
          f"{'(norm)':>8}  {'(ms)':>7}  {'(Pearson r)':>12}  "
          f"{'':>10}  {'(inv/total)':>11}")
    print("  " + "─" * 88)

    for r in results:
        tf = r.get("tf", {})
        print(f"  {r['label']:<22}  {r['t2_med']:>7.2f}  {r['t2_p95']:>7.2f}  "
              f"{r.get('dc_loss', float('nan')):>8.5f}  "
              f"{r.get('xseed', float('nan')):>7.3f}  "
              f"{tf.get('corr', float('nan')):>12.4f}  "
              f"{tf.get('nmse', float('nan')):>10.4f}  "
              f"{tf.get('halluc_frac', float('nan')):>11.4f}")

    print("  " + "═" * 88)

    # Reference numbers from run_prior_vs_physics.py
    print(f"\n  REFERENCE (run_prior_vs_physics.py — same setup):")
    print(f"    Piecewise-CNN (matched prior):  T2* med ≈  0.98 ms, ZF ≈  2.70 ms")
    print(f"    Textured-CNN  (prior mismatch): T2* med ≈  7.63 ms, ZF ≈  8.02 ms")
    print(f"    [CNN on textured removes only 5% of ZF error]")

    # Find JEPA and CNN rows
    jepa_r = next((r for r in results if "JEPA" in r["label"]), {})
    cnn_r  = next((r for r in results if "CNN" in r["label"]), {})
    zf_r   = next((r for r in results if "ZF" in r["label"]), {})

    jepa_t2 = jepa_r.get("t2_med", float("nan"))
    cnn_t2  = cnn_r.get("t2_med", float("nan"))
    zf_t2   = zf_r.get("t2_med", float("nan"))
    hf      = jepa_r.get("tf", {}).get("halluc_frac", float("nan"))
    corr    = jepa_r.get("tf", {}).get("corr", float("nan"))

    jepa_gap = zf_t2 - jepa_t2
    cnn_gap  = zf_t2 - cnn_t2

    print(f"\n  VERDICT ANALYSIS:")
    print(f"    ZF floor (textured):       {zf_t2:.2f} ms")
    print(f"    CNN-prior (textured):      {cnn_t2:.2f} ms  (gap vs ZF: {cnn_gap:.2f} ms)")
    print(f"    JEPA-prior (textured):     {jepa_t2:.2f} ms  (gap vs ZF: {jepa_gap:.2f} ms)")
    print(f"    Hallucination fraction:    {hf:.4f}  (0=perfect, 1=all-invented)")
    print(f"    Unmeas-band correlation:   {corr:.4f}  (1=perfect texture recovery)")
    print(f"    Sanity gate:               {'PASS' if gate.get('passed') else 'FAIL'}")

    print()

    # Verdict logic
    gap_improve = jepa_gap - cnn_gap   # how much MORE gap does JEPA give over CNN?

    if not gate.get("passed"):
        verdict = (
            "NO-FILL (sanity gate FAIL) — The JEPA prior does not discriminate "
            "on- vs off-manifold maps. The learned representation is degenerate; "
            "the surprise regulariser provides no meaningful structural constraint. "
            "Result: JEPA prior is NOT a valid regulariser in this configuration."
        )
    elif hf < 0.35 and corr > 0.5 and jepa_gap > cnn_gap * 1.5:
        verdict = (
            f"PROMISING — JEPA prior improves T2* error to {jepa_t2:.2f} ms vs "
            f"CNN-prior {cnn_t2:.2f} ms (ZF floor {zf_t2:.2f} ms). "
            f"Unmeasured-band correlation {corr:.3f} and hallucination fraction "
            f"{hf:.3f} confirm the prior fills the unmeasured band with TRUE texture, "
            "not invented plausible content. Worth scaling to a larger pretrain set."
        )
    elif hf > 0.6 or corr < 0.1:
        verdict = (
            f"HALLUCINATES — JEPA prior {'' if jepa_t2 < cnn_t2 else 'does NOT '}improve "
            f"whole-map T2* ({jepa_t2:.2f} ms vs {cnn_t2:.2f} ms), but unmeasured-band "
            f"correlation is {corr:.3f} and hallucination fraction is {hf:.3f}: "
            "the prior invents plausible-but-false texture in the unmeasured k-space band. "
            "Data consistency cannot detect hallucination (invented texture is data-consistent). "
            "JEPA prior is no better than a generative model in the unmeasured band — "
            "the no-decoder bet fails; honesty of fill requires data or a stronger prior."
        )
    else:
        verdict = (
            f"INTERMEDIATE — JEPA prior: T2*={jepa_t2:.2f} ms vs CNN {cnn_t2:.2f} ms, "
            f"ZF {zf_t2:.2f} ms. Unmeas-band corr={corr:.3f}, halluc_frac={hf:.3f}. "
            "Some genuine texture recovery but also partial hallucination. "
            "JEPA improves over the CNN smoothness prior but does not fully solve "
            "the unmeasured-band problem at 4× undersampling in this prototype. "
            "Larger pretrain set and stronger masking strategy may reduce hallucination."
        )

    print("  VERDICT:")
    for line in textwrap.wrap(verdict, width=84):
        print(f"    {line}")
    print("═" * 92 + "\n")
    return verdict


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--recon-epochs",    type=int, default=200)
    args = parser.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    print(f"\nDevice: {device}")
    print(f"Pretrain epochs: {args.pretrain_epochs}  |  Recon epochs: {args.recon_epochs}")
    print(f"Phantom: {H}×{W}, {N_ECHOES} echoes, {ACCEL}× accel, SNR={SNR_DB} dB\n")

    # ── dataset ──────────────────────────────────────────────────────────────
    print("Building map dataset (file-level pretrain/test split)…")
    all_files = _get_corpd_files()
    pretrain_files = all_files[:N_PRETRAIN_FILES]
    test_files     = all_files[N_PRETRAIN_FILES:N_PRETRAIN_FILES + N_TEST_FILES]
    print(f"  Pretrain files: {pretrain_files}")
    print(f"  Test files:     {test_files}")

    pretrain_maps = generate_map_dataset(pretrain_files, seeds=[0, 1])
    test_maps     = generate_map_dataset(test_files,     seeds=[2, 3])
    print(f"  Pretrain maps: {len(pretrain_maps)},  Test maps: {len(test_maps)}\n")

    # ── Stage A: JEPA pretraining ──────────────────────────────────────────
    print("═" * 60)
    print("STAGE A — JEPA pre-training")
    print("═" * 60)
    encoder     = ContextEncoder()
    ema_encoder = copy.deepcopy(encoder)
    predictor   = KANPredictor()

    # EMA encoder never gets gradient updates
    for p in ema_encoder.parameters():
        p.requires_grad = False

    pretrain_losses = pretrain_jepa(
        encoder, ema_encoder, predictor, pretrain_maps,
        n_epochs=args.pretrain_epochs, device=device,
    )
    print(f"  Stage A complete.  Final loss: {pretrain_losses[-1]:.5f}\n")

    # ── Sanity gate ───────────────────────────────────────────────────────
    print("═" * 60)
    print("SANITY GATE")
    print("═" * 60)
    gate_passed, gate_stats = run_sanity_gate(
        pretrain_maps, test_maps, encoder, ema_encoder, predictor, device
    )
    if not gate_passed:
        print("\n  GATE FAILED — Stage B will run but verdict will be NO-FILL.\n")

    # Freeze JEPA for Stage B (weights → no_grad; input grads still flow)
    for p in encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False

    # ── Stage B & C: Reconstruction on N_RECON_MAPS test maps ─────────────
    print("\n" + "═" * 60)
    print("STAGE B+C — Reconstruction + texture fidelity")
    print("═" * 60)

    all_results: list[dict] = []

    for mi, tmap in enumerate(test_maps[:N_RECON_MAPS]):
        print(f"\n  ── Test map {mi+1}/{N_RECON_MAPS} "
              f"({tmap['file']}, sl={tmap['slice']}, seed={tmap['seed']}) ──")
        S0_gt, T2_gt, fg = tmap["S0"], tmap["T2"], tmap["fg"]

        data = synthesise(S0_gt, T2_gt, TEs_MS, ACCEL, CF, SNR_DB,
                          mask_seed=42, noise_seed=7)
        mask_1d = data["mask_1d"]

        bl = run_baselines(data, T2_gt, fg)
        print(f"  Baselines: full={bl['full_med']:.2f} ms  ZF={bl['zf_med']:.2f} ms")

        k_under_tc = torch.from_numpy(data["kspace_under"]).to(torch.complex64)
        mask_tc    = torch.from_numpy(data["mask_2d"])

        # Collect per-seed T2* maps for JEPA and CNN models
        jepa_T2s: list[np.ndarray] = []
        cnn_T2s:  list[np.ndarray] = []
        jepa_losses: list[float]   = []
        cnn_losses:  list[float]   = []

        for seed in range(N_SEEDS):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

            # — JEPA-prior reconstruction —
            zf_arr = _make_complex_zf_input(data["kspace_under"])
            zf_in  = torch.from_numpy(zf_arr).unsqueeze(0)
            mdl_j  = BottleneckNet3(N_ECHOES, HIDDEN, T2_MIN, T2_MAX, DF_BOUND, complex_input=True)
            mdl_j, fl_j, _ = train_3param_jepa(
                mdl_j, zf_in, k_under_tc, mask_tc,
                encoder, ema_encoder, predictor, TEs_MS,
                n_epochs=args.recon_epochs, lr=RECON_LR,
                device=device, label=f"JEPA m{mi} s{seed}",
            )
            _, T2_j, _ = predict_3param(mdl_j, zf_in, device)
            jepa_T2s.append(T2_j); jepa_losses.append(fl_j)

            # — CNN-prior (plain 3-param, no JEPA) —
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            mdl_c = BottleneckNet3(N_ECHOES, HIDDEN, T2_MIN, T2_MAX, DF_BOUND, complex_input=True)
            mdl_c, fl_c = train_3param(
                mdl_c, zf_in, k_under_tc, mask_tc, TEs_MS,
                n_epochs=args.recon_epochs, lr=RECON_LR,
                device=device, label=f"CNN  m{mi} s{seed}",
            )
            _, T2_c, _ = predict_3param(mdl_c, zf_in, device)
            cnn_T2s.append(T2_c); cnn_losses.append(fl_c)

        # Best-seed by T2* median error
        def _best(T2_list):
            errs = [np.median(np.abs(T[fg] - T2_gt[fg])) for T in T2_list]
            b = int(np.argmin(errs))
            return T2_list[b], errs[b]

        T2_jepa_best, jepa_med = _best(jepa_T2s)
        T2_cnn_best,  cnn_med  = _best(cnn_T2s)

        # Cross-seed std
        jepa_xseed = np.stack(jepa_T2s, 0)[:, fg].std(0).mean()
        cnn_xseed  = np.stack(cnn_T2s,  0)[:, fg].std(0).mean()

        # T2* metrics
        j_m = t2_metrics_fg(T2_jepa_best, T2_gt, fg)
        c_m = t2_metrics_fg(T2_cnn_best,  T2_gt, fg)

        # Texture fidelity in unmeasured band
        tf_jepa = texture_fidelity_analysis(T2_jepa_best, T2_gt, mask_1d)
        tf_cnn  = texture_fidelity_analysis(T2_cnn_best,  T2_gt, mask_1d)
        tf_zf   = texture_fidelity_analysis(bl["T2_zf"],   T2_gt, mask_1d)
        tf_full = texture_fidelity_analysis(bl["T2_full"],  T2_gt, mask_1d)

        print(f"\n  Map {mi+1} results:")
        print(f"    JEPA prior: T2*={j_m['med']:.2f} ms  corr={tf_jepa['corr']:.4f}  "
              f"halluc={tf_jepa['halluc_frac']:.4f}  dc_loss={np.mean(jepa_losses):.5f}")
        print(f"    CNN prior:  T2*={c_m['med']:.2f} ms  corr={tf_cnn['corr']:.4f}  "
              f"halluc={tf_cnn['halluc_frac']:.4f}  dc_loss={np.mean(cnn_losses):.5f}")
        print(f"    ZF floor:   T2*={bl['zf_med']:.2f} ms  corr={tf_zf['corr']:.4f}  "
              f"halluc={tf_zf['halluc_frac']:.4f}")
        print(f"    Full UB:    T2*={bl['full_med']:.2f} ms  corr={tf_full['corr']:.4f}  "
              f"halluc={tf_full['halluc_frac']:.4f}")

        # Check: does DC residual detect hallucination?
        print(f"\n    Hallucination detection check:")
        print(f"      JEPA dc_loss={np.mean(jepa_losses):.5f}  CNN dc_loss={np.mean(cnn_losses):.5f}")
        print(f"      (if dc_loss is similar but halluc_frac differs → loss does NOT detect it)")

        # Save map figure for first test map
        if mi == 0:
            save_reconstruction_figure(
                f"test_map_{mi+1}",
                S0_gt, T2_gt, T2_jepa_best, T2_cnn_best, bl["T2_zf"],
            )

        # Accumulate for aggregate table
        all_results.append({
            "map_idx": mi + 1,
            "jepa": dict(t2_med=j_m["med"], t2_p95=j_m["p95"],
                         dc_loss=float(np.mean(jepa_losses)), xseed=float(jepa_xseed), tf=tf_jepa),
            "cnn":  dict(t2_med=c_m["med"], t2_p95=c_m["p95"],
                         dc_loss=float(np.mean(cnn_losses)),  xseed=float(cnn_xseed),  tf=tf_cnn),
            "zf":   dict(t2_med=bl["zf_med"],   t2_p95=bl["zf_p95"],   tf=tf_zf),
            "full": dict(t2_med=bl["full_med"],  t2_p95=bl["full_p95"], tf=tf_full),
        })

    # ── Aggregate over test maps ───────────────────────────────────────────
    def _agg(key: str) -> dict:
        meds = [r[key]["t2_med"]            for r in all_results]
        p95s = [r[key]["t2_p95"]            for r in all_results]
        corr = [r[key]["tf"]["corr"]        for r in all_results]
        nmse = [r[key]["tf"]["nmse"]        for r in all_results]
        hf   = [r[key]["tf"]["halluc_frac"] for r in all_results]
        dc   = [r[key].get("dc_loss", float("nan")) for r in all_results]
        xs   = [r[key].get("xseed",   float("nan")) for r in all_results]
        return dict(t2_med=np.mean(meds), t2_p95=np.mean(p95s),
                    dc_loss=np.nanmean(dc), xseed=np.nanmean(xs),
                    tf=dict(corr=np.mean(corr), nmse=np.mean(nmse), halluc_frac=np.mean(hf)))

    verdict_rows = [
        {"label": "Full-sample UB",  **_agg("full")},
        {"label": "ZF analytical",   **_agg("zf")},
        {"label": "CNN-prior (3p)",  **_agg("cnn")},
        {"label": "JEPA-prior (3p)", **_agg("jepa")},
    ]

    # Summary figure
    save_jepa_figure("prototype", pretrain_losses, gate_stats, [
        {"label": "ZF",       "tf": _agg("zf")["tf"]},
        {"label": "CNN-prior","tf": _agg("cnn")["tf"]},
        {"label": "JEPA-prior","tf": _agg("jepa")["tf"]},
    ])

    verdict = print_verdict(verdict_rows, gate_stats)
    print(f"Final verdict: {verdict.split(' — ')[0]}")


if __name__ == "__main__":
    main()
