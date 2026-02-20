"""Full-scale MRI data loader for fastMRI dataset.

Supports both singlecoil and multicoil knee/brain data with variable
acceleration factors and data normalization.
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def create_undersampling_mask(
    shape: Tuple[int, int],
    acceleration: int = 4,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Create 1D equispaced undersampling mask for k-space.
    
    Args:
        shape: (height, width) of k-space
        acceleration: Acceleration factor (4 or 8 for fastMRI)
        center_fraction: Fraction of center k-space to fully sample
        seed: Random seed for reproducibility (not used for equispaced)
        
    Returns:
        Binary mask tensor [H, W], 1 = sampled, 0 = not sampled
    """
    h, w = shape
    mask = torch.zeros(h, w)
    
    # Calculate center region
    num_center_lines = int(w * center_fraction)
    center_start = (w - num_center_lines) // 2
    center_end = center_start + num_center_lines
    
    # Keep center fully sampled
    mask[:, center_start:center_end] = 1.0
    
    # Equispaced sampling outside center
    num_remaining = w - num_center_lines
    num_samples_outside = num_remaining // acceleration
    
    # Left side sampling
    left_width = center_start
    if left_width > 0 and num_samples_outside > 0:
        left_count = max(1, num_samples_outside // 2)
        left_indices = torch.linspace(0, left_width - 1, left_count, dtype=torch.long)
        mask[:, left_indices] = 1.0
    
    # Right side sampling
    right_width = w - center_end
    if right_width > 0 and num_samples_outside > 0:
        right_count = max(1, num_samples_outside - (num_samples_outside // 2))
        right_indices = torch.linspace(center_end, w - 1, right_count, dtype=torch.long)
        mask[:, right_indices] = 1.0
    
    return mask


def create_random_undersampling_mask(
    shape: Tuple[int, int],
    acceleration: int = 4,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Create random 1D undersampling mask for k-space.
    
    Args:
        shape: (height, width) of k-space
        acceleration: Acceleration factor
        center_fraction: Fraction of center k-space to fully sample
        seed: Random seed for reproducibility
        
    Returns:
        Binary mask tensor [H, W], 1 = sampled, 0 = not sampled
    """
    h, w = shape
    mask = torch.zeros(h, w)
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Calculate center region
    num_center_lines = int(w * center_fraction)
    center_start = (w - num_center_lines) // 2
    center_end = center_start + num_center_lines
    
    # Keep center fully sampled
    mask[:, center_start:center_end] = 1.0
    
    # Random sampling outside center
    num_remaining = w - num_center_lines
    num_samples_outside = num_remaining // acceleration
    
    # Left side
    left_width = center_start
    if left_width > 0 and num_samples_outside > 0:
        left_count = max(1, num_samples_outside // 2)
        left_indices = torch.randperm(left_width)[:left_count]
        mask[:, left_indices] = 1.0
    
    # Right side
    right_width = w - center_end
    if right_width > 0 and num_samples_outside > 0:
        right_count = max(1, num_samples_outside - (num_samples_outside // 2))
        right_indices = center_end + torch.randperm(right_width)[:right_count]
        mask[:, right_indices] = 1.0
    
    return mask


def apply_mask(
    kspace: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply undersampling mask to k-space.
    
    Args:
        kspace: K-space tensor [2, H, W] or [num_coils, 2, H, W]
        mask: Binary mask [H, W]
        
    Returns:
        Masked k-space with same shape as input
    """
    if kspace.dim() == 3:
        # Singlecoil: [2, H, W]
        return kspace * mask.unsqueeze(0)
    elif kspace.dim() == 4:
        # Multicoil: [num_coils, 2, H, W]
        return kspace * mask.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected k-space shape: {kspace.shape}")


def coil_combine_rss(kspace: torch.Tensor) -> torch.Tensor:
    """Combine multi-coil data using Root Sum of Squares (RSS).
    
    Args:
        kspace: Multi-coil k-space [num_coils, 2, H, W]
        
    Returns:
        Combined magnitude image [H, W]
    """
    num_coils = kspace.shape[0]
    
    # Convert to complex
    complex_kspace = torch.view_as_complex(
        kspace.permute(0, 2, 3, 1).contiguous()
    )  # [num_coils, H, W]
    
    # IFFT2 per coil
    coil_images = torch.fft.ifft2(torch.fft.ifftshift(complex_kspace, dim=(-2, -1)), dim=(-2, -1))
    coil_images = torch.fft.fftshift(coil_images, dim=(-2, -1))
    
    # RSS combination
    rss_image = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0))
    
    return rss_image


def kspace_to_image(kspace: torch.Tensor) -> torch.Tensor:
    """Convert single-coil k-space to magnitude image.
    
    Args:
        kspace: K-space tensor [2, H, W] with real/imag channels
        
    Returns:
        Magnitude image [H, W]
    """
    # Convert real/imag to complex
    complex_kspace = torch.view_as_complex(kspace.permute(1, 2, 0).contiguous())
    
    # IFFT2
    image = torch.fft.ifft2(torch.fft.ifftshift(complex_kspace))
    image = torch.fft.fftshift(image)
    
    # Magnitude
    return torch.abs(image)


def normalize_kspace(
    kspace: torch.Tensor,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalize k-space data (per-slice zero mean, unit std).
    
    Args:
        kspace: K-space tensor [2, H, W] or [num_coils, 2, H, W]
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (normalized_kspace, stats_dict)
        stats_dict contains 'mean', 'std', 'max' for unnormalization
    """
    # Compute stats over all dimensions
    mean = kspace.mean()
    std = kspace.std()
    max_val = kspace.abs().max()
    
    # Normalize
    normalized = (kspace - mean) / (std + eps)
    
    stats = {
        'mean': mean.item(),
        'std': std.item(),
        'max': max_val.item(),
    }
    
    return normalized, stats


def create_normalized_coords(shape: Tuple[int, int]) -> torch.Tensor:
    """Create normalized coordinate grid from -1 to 1.
    
    Args:
        shape: (H, W) shape
        
    Returns:
        Coordinate tensor [H, W, 2] with (x, y) normalized to [-1, 1]
    """
    h, w = shape
    
    # Create grid from -1 to 1
    y_coords = torch.linspace(-1, 1, h)
    x_coords = torch.linspace(-1, 1, w)
    
    # Create meshgrid
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Stack to get [H, W, 2] with (x, y) coordinates
    coords = torch.stack([xx, yy], dim=-1)
    
    return coords


class FastMRIDataset(Dataset):
    """Full-scale fastMRI dataset for MRI reconstruction.
    
    Supports both singlecoil and multicoil data with variable acceleration.
    
    Args:
        root_dir: Root directory containing HDF5 files
        split: Data split ('train', 'val', or 'test')
        coil_type: 'singlecoil' or 'multicoil'
        acceleration: Acceleration factor (4 or 8)
        center_fraction: Fraction of center k-space to keep
        use_random_mask: If True, use random sampling; else equispaced
        normalize: If True, apply per-slice normalization
        slice_filter: Optional callable to filter slices
        
    Returns:
        Dictionary with:
            - 'kspace_under': Undersampled k-space [2, H, W] or [num_coils, 2, H, W]
            - 'kspace_full': Fully sampled k-space (normalized)
            - 'mask': Sampling mask [H, W]
            - 'target': Target magnitude image [H, W]
            - 'coords': Normalized coordinates [H, W, 2]
            - 'stats': Normalization statistics dict
            - 'filename': Source filename
            - 'slice_idx': Slice index in volume
    """
    
    # Center fractions for different accelerations (fastMRI standard)
    CENTER_FRACTIONS = {
        4: 0.08,
        8: 0.04,
    }
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        coil_type: str = 'singlecoil',
        acceleration: int = 4,
        center_fraction: Optional[float] = None,
        use_random_mask: bool = False,
        normalize: bool = True,
        slice_filter: Optional[Callable[[torch.Tensor], bool]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.coil_type = coil_type
        self.acceleration = acceleration
        self.use_random_mask = use_random_mask
        self.normalize = normalize
        self.slice_filter = slice_filter
        
        # Set center fraction
        if center_fraction is None:
            self.center_fraction = self.CENTER_FRACTIONS.get(acceleration, 0.08)
        else:
            self.center_fraction = center_fraction
        
        # Validate inputs
        if coil_type not in ['singlecoil', 'multicoil']:
            raise ValueError(f"coil_type must be 'singlecoil' or 'multicoil', got {coil_type}")
        
        # Scan for HDF5 files
        self.file_list = self._scan_files()
        
        # Build slice index
        self.slice_indices = self._build_slice_index()
        
        # Pre-compute coordinates (will be resized per sample)
        self.base_coords = None
        
        print(f"FastMRIDataset initialized:")
        print(f"  Split: {split}")
        print(f"  Coil type: {coil_type}")
        print(f"  Acceleration: {acceleration}x")
        print(f"  Center fraction: {self.center_fraction}")
        print(f"  Files: {len(self.file_list)}")
        print(f"  Slices: {len(self.slice_indices)}")
    
    def _scan_files(self) -> List[Path]:
        """Scan for HDF5 files in root directory."""
        # Look for HDF5 files recursively
        h5_files = list(self.root_dir.rglob('*.h5'))
        
        if len(h5_files) == 0:
            raise RuntimeError(f"No HDF5 files found in {self.root_dir}")
        
        # Filter by coil type if directory structure allows
        coil_files = [f for f in h5_files if self.coil_type in f.name or self.coil_type in str(f)]
        
        if len(coil_files) > 0:
            return sorted(coil_files)
        
        return sorted(h5_files)
    
    def _build_slice_index(self) -> List[Tuple[Path, int]]:
        """Build index of all valid slices."""
        slice_indices = []
        
        for h5_file in self.file_list:
            try:
                with h5py.File(h5_file, 'r') as f:
                    num_slices = f['kspace'].shape[0]
                    
                    for slice_idx in range(num_slices):
                        # Optional: filter slices based on criteria
                        if self.slice_filter is not None:
                            kspace_slice = f['kspace'][slice_idx]
                            kspace_tensor = self._numpy_to_tensor(kspace_slice)
                            
                            if not self.slice_filter(kspace_tensor):
                                continue
                        
                        slice_indices.append((h5_file, slice_idx))
                        
            except Exception as e:
                print(f"Warning: Could not read {h5_file}: {e}")
                continue
        
        return slice_indices
    
    def _numpy_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor with proper dtype."""
        if arr.dtype == np.complex64 or arr.dtype == np.complex128:
            # Complex data
            real = torch.from_numpy(arr.real)
            imag = torch.from_numpy(arr.imag)
            return torch.stack([real, imag], dim=0).float()
        elif arr.shape[-1] == 2:
            # Last dimension is real/imag
            return torch.from_numpy(arr).permute(-1, *range(len(arr.shape) - 1)).float()
        else:
            return torch.from_numpy(arr).float()
    
    def __len__(self) -> int:
        return len(self.slice_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int, Dict]]:
        """Get a single sample."""
        h5_file, slice_idx = self.slice_indices[idx]
        
        # Load k-space
        with h5py.File(h5_file, 'r') as f:
            kspace_slice = f['kspace'][slice_idx]
            kspace_full = self._numpy_to_tensor(kspace_slice)
        
        # Get shape
        if self.coil_type == 'multicoil':
            num_coils, _, h, w = kspace_full.shape
        else:
            _, h, w = kspace_full.shape
        
        # Create mask
        mask_fn = create_random_undersampling_mask if self.use_random_mask else create_undersampling_mask
        mask = mask_fn(
            (h, w),
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            seed=idx if self.use_random_mask else None,
        )
        
        # Create coords
        coords = create_normalized_coords((h, w))
        
        # Normalize if requested
        stats = {'mean': 0.0, 'std': 1.0, 'max': 1.0}
        if self.normalize:
            kspace_full, stats = normalize_kspace(kspace_full)
        
        # Apply mask
        kspace_under = apply_mask(kspace_full, mask)
        
        # Compute target
        if self.coil_type == 'multicoil':
            target = coil_combine_rss(kspace_full)
        else:
            target = kspace_to_image(kspace_full)
        
        return {
            'kspace_under': kspace_under,
            'kspace_full': kspace_full,
            'mask': mask,
            'target': target,
            'coords': coords,
            'stats': stats,
            'filename': h5_file.name,
            'slice_idx': slice_idx,
        }


def create_dataloaders(
    root_dir: Union[str, Path],
    coil_type: str = 'singlecoil',
    acceleration: int = 4,
    batch_size: int = 1,
    num_workers: int = 4,
    use_random_mask: bool = False,
    normalize: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders.
    
    Args:
        root_dir: Root directory containing train/val/test subdirectories
        coil_type: 'singlecoil' or 'multicoil'
        acceleration: Acceleration factor
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_random_mask: Use random undersampling
        normalize: Apply normalization
        
    Returns:
        Dictionary of dataloaders for each split
    """
    root_dir = Path(root_dir)
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = root_dir / split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping {split}")
            continue
        
        dataset = FastMRIDataset(
            root_dir=split_dir,
            split=split,
            coil_type=coil_type,
            acceleration=acceleration,
            use_random_mask=use_random_mask if split == 'train' else False,
            normalize=normalize,
        )
        
        shuffle = (split == 'train')
        
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return dataloaders


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Test FastMRIDataset')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory with HDF5 files')
    parser.add_argument('--coil_type', type=str, default='singlecoil', choices=['singlecoil', 'multicoil'])
    parser.add_argument('--acceleration', type=int, default=4, choices=[4, 8])
    parser.add_argument('--output_dir', type=str, default='demo', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    dataset = FastMRIDataset(
        root_dir=args.root_dir,
        split='train',
        coil_type=args.coil_type,
        acceleration=args.acceleration,
        use_random_mask=False,
        normalize=True,
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Visualize samples
    num_to_viz = min(args.num_samples, len(dataset))
    
    for i in range(num_to_viz):
        sample = dataset[i]
        
        print(f"\nSample {i}:")
        print(f"  Filename: {sample['filename']}")
        print(f"  Slice: {sample['slice_idx']}")
        for key in ['kspace_under', 'kspace_full', 'mask', 'target', 'coords']:
            value = sample[key]
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.4f}, {value.max():.4f}]")
        print(f"  Stats: {sample['stats']}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        kspace_full_mag = torch.sqrt(sample['kspace_full'][0]**2 + sample['kspace_full'][1]**2)
        kspace_under_mag = torch.sqrt(sample['kspace_under'][0]**2 + sample['kspace_under'][1]**2)
        
        kspace_full_log = torch.log(kspace_full_mag + 1e-10)
        kspace_under_log = torch.log(kspace_under_mag + 1e-10)
        
        axes[0, 0].imshow(kspace_full_log.numpy(), cmap='gray')
        axes[0, 0].set_title('Full K-space (log)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(kspace_under_log.numpy(), cmap='gray')
        axes[0, 1].set_title(f'Undersampled (R={args.acceleration})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sample['mask'].numpy(), cmap='gray')
        axes[0, 2].set_title('Mask')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(sample['target'].numpy(), cmap='gray')
        axes[1, 0].set_title('Target')
        axes[1, 0].axis('off')
        
        # Zero-filled recon
        kspace_under_complex = torch.view_as_complex(
            sample['kspace_under'].permute(1, 2, 0).contiguous()
        )
        zero_filled = torch.abs(
            torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace_under_complex)))
        )
        axes[1, 1].imshow(zero_filled.numpy(), cmap='gray')
        axes[1, 1].set_title('Zero-filled')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(sample['coords'][:, :, 0].numpy(), cmap='coolwarm')
        axes[1, 2].set_title('X Coords')
        axes[1, 2].axis('off')
        
        plt.suptitle(f"{sample['filename']} - Slice {sample['slice_idx']}")
        plt.tight_layout()
        
        save_path = os.path.join(args.output_dir, f'sample_{i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved to {save_path}")
    
    print(f"\nVisualized {num_to_viz} samples")
