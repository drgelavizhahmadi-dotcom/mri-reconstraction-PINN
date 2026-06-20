#!/usr/bin/env python3
"""
Pipeline fix — STEP 1 + changes A → D.

Runs each change on a fixed 200-slice training subset / 50-slice val subset
for 2 epochs, then prints a running metric table.

Usage:
    python scripts/run_fixes.py --data data/singlecoil_val
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mhwf_pikan.data.fastmri_v2 import FastMRIDatasetV2
from mhwf_pikan.core.fno_image import FNOImageReconstructor


# ════════════════════════════════════════════════════════════════════════════
#  FFT helpers (torch, MPS-compatible)
# ════════════════════════════════════════════════════════════════════════════

def ifft2c_torch(kspace_2ch: torch.Tensor) -> torch.Tensor:
    """Centered IFFT2.  kspace_2ch: [B,2,H,W] real/imag → [B,2,H,W] complex image."""
    kc = torch.view_as_complex(kspace_2ch.permute(0, 2, 3, 1).contiguous())  # [B,H,W]
    img = torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(kc, dim=(-2, -1)),
            dim=(-2, -1), norm="ortho",
        ),
        dim=(-2, -1),
    )
    return torch.view_as_real(img).permute(0, 3, 1, 2).contiguous()  # [B,2,H,W]


def fft2c_torch(image_2ch: torch.Tensor) -> torch.Tensor:
    """Centered FFT2.  image_2ch: [B,2,H,W] real/imag → [B,2,H,W] k-space."""
    ic = torch.view_as_complex(image_2ch.permute(0, 2, 3, 1).contiguous())  # [B,H,W]
    ks = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(ic, dim=(-2, -1)),
            dim=(-2, -1), norm="ortho",
        ),
        dim=(-2, -1),
    )
    return torch.view_as_real(ks).permute(0, 3, 1, 2).contiguous()  # [B,2,H,W]


def apply_dc(pred_2ch: torch.Tensor,
             kspace_under: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """
    Hard data consistency.
    k_dc = mask * k_measured + (1-mask) * fft(pred)
    Returns refined image [B, 2, H, W].
    """
    k_pred = fft2c_torch(pred_2ch)
    m = mask.unsqueeze(1)                  # [B,1,H,W]
    k_dc = m * kspace_under + (1 - m) * k_pred
    return ifft2c_torch(k_dc)             # [B,2,H,W]


# ════════════════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════════════════

def psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    mse = np.mean((pred - target) ** 2)
    return float(20 * np.log10(max_val / (np.sqrt(mse) + 1e-11)))


def nmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sum((pred - target) ** 2) / (np.sum(target ** 2) + 1e-11))


def ssim_simple(pred: np.ndarray, target: np.ndarray,
                data_range: float = 1.0) -> float:
    """Lightweight SSIM (avoids scikit-image dependency)."""
    try:
        from skimage.metrics import structural_similarity as _ssim
        return float(_ssim(pred, target, data_range=data_range))
    except ImportError:
        pass
    # Manual 11×11 Gaussian-window SSIM
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
    t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
    k = 11
    pad = k // 2
    mu_p = F.avg_pool2d(p, k, stride=1, padding=pad)
    mu_t = F.avg_pool2d(t, k, stride=1, padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sg_p2 = F.avg_pool2d(p ** 2, k, stride=1, padding=pad) - mu_p2
    sg_t2 = F.avg_pool2d(t ** 2, k, stride=1, padding=pad) - mu_t2
    sg_pt = F.avg_pool2d(p * t, k, stride=1, padding=pad) - mu_pt
    num = (2 * mu_pt + C1) * (2 * sg_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sg_p2 + sg_t2 + C2)
    return float((num / (den + 1e-11)).mean())


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Zero-filled baseline
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_zerofilled(loader: DataLoader) -> dict[str, float]:
    """Compute metrics for zero-filled IFFT baseline."""
    all_psnr, all_nmse, all_ssim = [], [], []
    for batch in loader:
        zf = batch["input"]          # [B,2,H,W]  zero-filled complex (normalised)
        target = batch["target"]     # [B,H,W]

        # Magnitude of zero-filled image
        zf_mag = torch.sqrt(zf[:, 0] ** 2 + zf[:, 1] ** 2)  # [B,H,W]
        zf_mag = torch.clamp(zf_mag, 0, 1)

        for i in range(zf_mag.shape[0]):
            p = zf_mag[i].cpu().numpy()
            t = target[i].cpu().numpy()
            all_psnr.append(psnr(p, t))
            all_nmse.append(nmse(p, t))
            all_ssim.append(ssim_simple(p, t))

    return {
        "PSNR":  float(np.mean(all_psnr)),
        "NMSE":  float(np.mean(all_nmse)),
        "SSIM":  float(np.mean(all_ssim)),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Generic train + eval
# ════════════════════════════════════════════════════════════════════════════

def train_eval(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn,
    n_epochs: int = 2,
    accum_steps: int = 1,
    use_dc: bool = False,
    label: str = "",
) -> dict[str, float]:
    """Train for n_epochs then evaluate. Returns metric dict."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    model.train()

    for epoch in range(n_epochs):
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        opt.zero_grad()
        for step, batch in enumerate(train_loader):
            inp    = batch["input"].to(device)    # [B,2,H,W]
            target = batch["target"].to(device)   # [B,H,W]
            kspace = batch["kspace"].to(device)
            mask   = batch["mask"].to(device)

            pred = model(inp)                     # [B,1,H,W] or [B,2,H,W]

            if use_dc:
                # pred [B,2,H,W] complex → DC → magnitude
                pred = apply_dc(pred, kspace, mask)
                pred_mag = torch.sqrt(pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2)
            else:
                pred_mag = pred if pred.shape[1] == 1 else \
                    torch.sqrt(pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2)

            loss = loss_fn(pred_mag, target.unsqueeze(1)) / accum_steps
            loss.backward()
            running_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            n_batches += 1
            if device.type == "mps" and step % 20 == 0:
                torch.mps.empty_cache()

        avg_loss = running_loss / n_batches
        print(f"  [{label}] epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}  "
              f"({time.time()-t0:.0f}s)")

    return evaluate(model, val_loader, device, use_dc)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_dc: bool = False,
) -> dict[str, float]:
    model.eval()
    all_psnr, all_nmse, all_ssim = [], [], []

    for batch in val_loader:
        inp    = batch["input"].to(device)
        target = batch["target"].to(device)
        kspace = batch["kspace"].to(device)
        mask   = batch["mask"].to(device)

        pred = model(inp)
        if use_dc:
            pred = apply_dc(pred, kspace, mask)

        pred_mag = torch.sqrt(pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2) \
                   if pred.shape[1] == 2 else pred
        pred_mag = torch.clamp(pred_mag.squeeze(1), 0, 1)

        if device.type == "mps":
            torch.mps.empty_cache()

        for i in range(pred_mag.shape[0]):
            p = pred_mag[i].cpu().numpy()
            t = target[i].cpu().numpy()
            all_psnr.append(psnr(p, t))
            all_nmse.append(nmse(p, t))
            all_ssim.append(ssim_simple(p, t))

    model.train()
    return {
        "PSNR":  float(np.mean(all_psnr)),
        "NMSE":  float(np.mean(all_nmse)),
        "SSIM":  float(np.mean(all_ssim)),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Print table
# ════════════════════════════════════════════════════════════════════════════

def print_table(rows: list[tuple[str, dict]]) -> None:
    header = f"{'Stage':<30} {'PSNR (dB)':>10} {'NMSE':>10} {'SSIM':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label, m in rows:
        print(f"{label:<30} {m['PSNR']:>10.2f} {m['NMSE']:>10.4f} {m['SSIM']:>8.4f}")
    print("=" * len(header) + "\n")


# ════════════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/singlecoil_val",
                        help="Path to directory with fastMRI .h5 files")
    parser.add_argument("--train_slices", type=int, default=200,
                        help="Number of training slices for quick experiments")
    parser.add_argument("--val_slices", type=int, default=50,
                        help="Number of val slices")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs per change (default 2 — signal only)")
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--n_fno", type=int, default=4)
    args = parser.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # ── datasets ──────────────────────────────────────────────────────────
    print("\nBuilding datasets …")
    train_ds = FastMRIDatasetV2(args.data, split="train",
                                max_slices=args.train_slices)
    val_ds   = FastMRIDatasetV2(args.data, split="val",
                                max_slices=args.val_slices)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=0)

    table: list[tuple[str, dict]] = []

    # ── STEP 1: zero-filled baseline ──────────────────────────────────────
    print("\n─── STEP 1: Zero-filled baseline ───")
    zf_metrics = eval_zerofilled(val_loader)
    print(f"  PSNR={zf_metrics['PSNR']:.2f} dB  "
          f"NMSE={zf_metrics['NMSE']:.4f}  "
          f"SSIM={zf_metrics['SSIM']:.4f}")
    table.append(("ZF baseline (IFFT magnitude)", zf_metrics))
    print_table(table)

    # ── Change A: image-domain input + correct normalisation ──────────────
    print("─── Change A: image-domain FNO, MSE loss ───")
    model_a = FNOImageReconstructor(args.modes, args.width, args.n_fno,
                                    output_complex=False)
    print(f"  Params: {FNOImageReconstructor.count_params(model_a):,}")
    m_a = train_eval(model_a, train_loader, val_loader, device,
                     loss_fn=nn.MSELoss(),
                     n_epochs=args.epochs, accum_steps=1,
                     use_dc=False, label="A")
    table.append(("A: image-domain + norm (MSE)", m_a))
    print_table(table)

    # ── Change B: switch MSE → L1 (+SSIM) ────────────────────────────────
    print("─── Change B: L1 loss ───")
    model_b = FNOImageReconstructor(args.modes, args.width, args.n_fno,
                                    output_complex=False)
    # Warm-start from Change A weights
    model_b.load_state_dict(model_a.state_dict())

    def l1_ssim_loss(pred: torch.Tensor, target: torch.Tensor,
                     alpha: float = 0.84) -> torch.Tensor:
        """alpha * (1-SSIM) + (1-alpha) * L1  — approximation."""
        l1 = F.l1_loss(pred, target)
        # Simple mean-field SSIM approximation
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        mu_p = F.avg_pool2d(pred,   11, 1, 5)
        mu_t = F.avg_pool2d(target, 11, 1, 5)
        sig_p = F.avg_pool2d(pred ** 2,    11, 1, 5) - mu_p ** 2
        sig_t = F.avg_pool2d(target ** 2,  11, 1, 5) - mu_t ** 2
        sig_pt= F.avg_pool2d(pred * target,11, 1, 5) - mu_p * mu_t
        ssim_map = ((2*mu_p*mu_t + C1)*(2*sig_pt + C2)) / \
                   ((mu_p**2 + mu_t**2 + C1)*(sig_p + sig_t + C2) + 1e-8)
        ssim_loss = 1 - ssim_map.mean()
        return alpha * ssim_loss + (1 - alpha) * l1

    m_b = train_eval(model_b, train_loader, val_loader, device,
                     loss_fn=l1_ssim_loss,
                     n_epochs=args.epochs, accum_steps=1,
                     use_dc=False, label="B")
    table.append(("B: L1 + SSIM loss", m_b))
    print_table(table)

    # ── Change C: complex output + data consistency ───────────────────────
    print("─── Change C: complex output + DC ───")
    model_c = FNOImageReconstructor(args.modes, args.width, args.n_fno,
                                    output_complex=True)
    # Copy lift + shared weights from B (lift layers match since input is same)
    # We can't copy the project layer (different out_ch), so only copy body
    model_c.lift.weight.data.copy_(model_b.lift.weight.data)
    model_c.lift.bias.data.copy_(model_b.lift.bias.data)
    for blk_c, blk_b in zip(model_c.blocks, model_b.blocks):
        blk_c.load_state_dict(blk_b.state_dict())

    m_c = train_eval(model_c, train_loader, val_loader, device,
                     loss_fn=l1_ssim_loss,
                     n_epochs=args.epochs, accum_steps=1,
                     use_dc=True, label="C")
    table.append(("C: complex out + DC", m_c))
    print_table(table)

    # ── Change D: gradient accumulation (effective batch 8) ──────────────
    print("─── Change D: grad accumulation × 8 ───")
    model_d = FNOImageReconstructor(args.modes, args.width, args.n_fno,
                                    output_complex=True)
    model_d.load_state_dict(model_c.state_dict())

    m_d = train_eval(model_d, train_loader, val_loader, device,
                     loss_fn=l1_ssim_loss,
                     n_epochs=args.epochs, accum_steps=8,
                     use_dc=True, label="D")
    table.append(("D: + grad accum ×8", m_d))

    # ── Final table ───────────────────────────────────────────────────────
    print("\n══════════════════ FINAL RESULTS ══════════════════")
    print_table(table)

    # Check if we beat ZF baseline
    zf_psnr = table[0][1]["PSNR"]
    final_psnr = table[-1][1]["PSNR"]
    if final_psnr > zf_psnr:
        delta = final_psnr - zf_psnr
        print(f"✓ Final model beats zero-filling by {delta:.2f} dB PSNR")
    else:
        delta = zf_psnr - final_psnr
        print(f"✗ Final model is {delta:.2f} dB BELOW zero-filling — check pipeline")

    print("\nDone.")


if __name__ == "__main__":
    main()
