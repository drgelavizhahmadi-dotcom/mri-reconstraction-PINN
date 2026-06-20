"""
FastMRI dataset V2 — correct normalization and image-domain input.

Key fixes vs V1:
  - Normalization: per-slice max(|RSS image|), same scale applied to both
    the zero-filled input and the target. This keeps zero-filled ≈ [0,1]
    and target in [0,1] with consistent physical meaning.
  - Centered IFFT: ifft2c = fftshift(ifft2(ifftshift(x))) matching fastMRI standard.
  - Input: zero-filled complex image [2,H,W] (real/imag), NOT raw k-space.
  - Returns undersampled k-space [2,H,W] and mask [H,W] for data consistency.
  - Mask is applied in k-space (column-wise 1-D equispaced + random outer lines).
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, Union


# ──────────────────────────── FFT helpers (numpy) ────────────────────────────

def ifft2c(kspace: np.ndarray) -> np.ndarray:
    """Centered IFFT2 (fastMRI convention). kspace: complex [...,H,W]."""
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1), norm="ortho",
        ),
        axes=(-2, -1),
    )


def fft2c(image: np.ndarray) -> np.ndarray:
    """Centered FFT2. image: complex [...,H,W]."""
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(image, axes=(-2, -1)),
            axes=(-2, -1), norm="ortho",
        ),
        axes=(-2, -1),
    )


# ──────────────────────────────── Dataset ────────────────────────────────────

class FastMRIDatasetV2(Dataset):
    """
    FastMRI singlecoil knee dataset with corrected normalization pipeline.

    Returns per __getitem__:
        'input'  : [2,H,W] float  — zero-filled complex image (real/imag), scale-normalized
        'target' : [H,W]   float  — fully-sampled magnitude image, scale-normalized ∈ [0,1]
        'kspace' : [2,H,W] float  — undersampled k-space (real/imag), scale-normalized
        'mask'   : [H,W]   float  — binary sampling mask (1=sampled)
        'scale'  : scalar  float  — normalization factor (max of full magnitude)

    Args:
        root_dir       : directory containing .h5 files
        split          : 'train' | 'val' | 'test'
        acceleration   : R-factor (default 4)
        center_fraction: fraction of center k-space lines always sampled (default 0.08)
        target_size    : spatial size after cropping (default 320×320)
        max_slices     : cap number of slices (useful for quick experiments)
        use_seed       : fix mask per slice for reproducibility
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        acceleration: float = 4.0,
        center_fraction: float = 0.08,
        target_size: Tuple[int, int] = (320, 320),
        max_slices: Optional[int] = None,
        use_seed: bool = True,
    ) -> None:
        assert split in ("train", "val", "test")
        self.root_dir = Path(root_dir)
        self.split = split
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.target_size = target_size
        self.use_seed = use_seed

        h5_files = sorted(self.root_dir.glob("*.h5"))
        assert h5_files, f"No .h5 files in {root_dir}"

        # Build (file_idx, slice_idx) index
        all_slices: list[tuple[int, int]] = []
        for fid, p in enumerate(h5_files):
            try:
                with h5py.File(p, "r") as f:
                    for sid in range(f["kspace"].shape[0]):
                        all_slices.append((fid, sid))
            except Exception as exc:
                print(f"Warning: skipping {p}: {exc}")

        n = len(all_slices)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if split == "train":
            slices = all_slices[:n_train]
        elif split == "val":
            slices = all_slices[n_train : n_train + n_val]
        else:
            slices = all_slices[n_train + n_val :]

        if max_slices is not None:
            slices = slices[:max_slices]

        self.h5_files = h5_files
        self.slices = slices
        self._mask_cache: dict = {}

        print(f"FastMRIDatasetV2 [{split}]: {len(self.slices)} slices "
              f"(acc={acceleration}, cf={center_fraction}, size={target_size})")

    def __len__(self) -> int:
        return len(self.slices)

    # ─────────────────────────── helpers ─────────────────────────────────────

    def _make_mask_1d(self, width: int, seed: int) -> np.ndarray:
        """1-D column mask: center block + random outer lines."""
        cache_key = (width, seed)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        mask = np.zeros(width, dtype=np.float32)
        n_center = round(width * self.center_fraction)
        c0 = (width - n_center) // 2
        mask[c0 : c0 + n_center] = 1.0

        outer = np.concatenate([np.arange(c0), np.arange(c0 + n_center, width)])
        n_outer = max(1, round(len(outer) / self.acceleration))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(outer, size=min(n_outer, len(outer)), replace=False)
        mask[chosen] = 1.0

        self._mask_cache[cache_key] = mask
        return mask

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        """Center-crop last two dims to target_size; zero-pad if smaller."""
        th, tw = self.target_size
        h, w = image.shape[-2], image.shape[-1]

        h0 = max(0, (h - th) // 2)
        w0 = max(0, (w - tw) // 2)
        out = image[..., h0 : h0 + th, w0 : w0 + tw]

        ch, cw = out.shape[-2], out.shape[-1]
        if ch < th or cw < tw:
            pad = [(0, 0)] * (out.ndim - 2) + [(0, th - ch), (0, tw - cw)]
            out = np.pad(out, pad)
        return out

    # ─────────────────────────── __getitem__ ─────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fid, sid = self.slices[idx]
        with h5py.File(self.h5_files[fid], "r") as f:
            raw = f["kspace"][sid]

        # Ensure complex64
        if np.iscomplexobj(raw):
            kspace_full = raw.astype(np.complex64)  # [H, W]
        else:
            kspace_full = (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)

        # Fully-sampled image (centered IFFT)
        image_full = ifft2c(kspace_full)          # [H, W] complex
        image_full = self._center_crop(image_full) # [th, tw] complex

        # Ground-truth magnitude + per-slice normalization scale
        mag_full = np.abs(image_full)              # [th, tw] real ≥ 0
        scale = float(mag_full.max()) + 1e-11
        target = (mag_full / scale).astype(np.float32)  # [th, tw] ∈ [0, 1]

        # Re-derive k-space from the cropped image (so mask/size match)
        kspace_cropped = fft2c(image_full)         # [th, tw] complex

        # Undersampling mask (column-wise, broadcast to [th, tw])
        th, tw = self.target_size
        seed = idx if self.use_seed else 0
        mask_1d = self._make_mask_1d(tw, seed)
        mask_2d = np.broadcast_to(mask_1d[None, :], (th, tw)).copy().astype(np.float32)

        # Undersampled k-space and zero-filled image
        kspace_under = (kspace_cropped * mask_2d).astype(np.complex64)
        image_zf = ifft2c(kspace_under)            # [th, tw] complex

        # Normalise by the same per-slice scale
        zf_norm = image_zf / scale                 # complex, values ≈ [-1, 1]
        ks_norm = kspace_under / scale             # complex

        return {
            "input": torch.from_numpy(
                np.stack([zf_norm.real, zf_norm.imag], axis=0)
            ).float(),                                              # [2,H,W]
            "target": torch.from_numpy(target).float(),            # [H,W]
            "kspace": torch.from_numpy(
                np.stack([ks_norm.real, ks_norm.imag], axis=0)
            ).float(),                                             # [2,H,W]
            "mask": torch.from_numpy(mask_2d).float(),             # [H,W]
            "scale": torch.tensor(scale, dtype=torch.float32),
        }
