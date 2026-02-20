"""Toy dataset for fastMRI singlecoil knee data."""

import os
from typing import Dict, Tuple

import h5py
import torch
from torch.utils.data import Dataset


def create_undersampling_mask(shape: Tuple[int, int], acceleration: int = 4, center_fraction: float = 0.08) -> torch.Tensor:
    """Create 1D equispaced undersampling mask for k-space.
    
    Args:
        shape: (height, width) of k-space
        acceleration: Acceleration factor
        center_fraction: Fraction of center k-space to keep
        
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
    if left_width > 0:
        left_indices = torch.linspace(0, left_width - 1, num_samples_outside // 2, dtype=torch.long)
        mask[:, left_indices] = 1.0
    
    # Right side sampling
    right_width = w - center_end
    if right_width > 0:
        right_indices = torch.linspace(center_end, w - 1, num_samples_outside // 2, dtype=torch.long)
        mask[:, right_indices] = 1.0
    
    return mask


def extract_center_patch(kspace: torch.Tensor, patch_size: int = 64) -> torch.Tensor:
    """Extract center patch from k-space.
    
    Args:
        kspace: Complex k-space tensor [H, W] (complex dtype) or [2, H, W] (real/imag)
        patch_size: Size of center patch to extract
        
    Returns:
        Center patch of size [patch_size, patch_size] or [2, patch_size, patch_size]
    """
    if kspace.dim() == 3:
        # Already in [2, H, W] format
        _, h, w = kspace.shape
        h_start = (h - patch_size) // 2
        w_start = (w - patch_size) // 2
        return kspace[:, h_start:h_start + patch_size, w_start:w_start + patch_size]
    else:
        # Complex tensor [H, W]
        h, w = kspace.shape
        h_start = (h - patch_size) // 2
        w_start = (w - patch_size) // 2
        return kspace[h_start:h_start + patch_size, w_start:w_start + patch_size]


def kspace_to_image(kspace: torch.Tensor) -> torch.Tensor:
    """Convert k-space to magnitude image.
    
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
    
    # Create meshgrid - note: meshgrid order is (x, y) for 'xy' indexing
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Stack to get [H, W, 2] with (x, y) coordinates
    coords = torch.stack([xx, yy], dim=-1)
    return coords


class ToyFastMRIDataset(Dataset):
    """Toy dataset for single fastMRI HDF5 file.
    
    Loads singlecoil knee data and extracts 64x64 center patches.
    
    Args:
        h5_path: Path to fastMRI HDF5 file
        patch_size: Size of center patch to extract (default: 64)
        acceleration: Undersampling acceleration factor (default: 4)
        center_fraction: Fraction of center k-space to keep (default: 0.08)
    """
    
    def __init__(
        self,
        h5_path: str,
        patch_size: int = 64,
        acceleration: int = 4,
        center_fraction: float = 0.08,
    ):
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        
        # Pre-compute mask and coords (same for all slices)
        self.mask = create_undersampling_mask(
            (patch_size, patch_size), acceleration, center_fraction
        )
        self.coords = create_normalized_coords((patch_size, patch_size))
        
        # Open file to get number of slices
        with h5py.File(h5_path, 'r') as f:
            kspace_dataset = f['kspace']
            self.num_slices = kspace_dataset.shape[0]
            self.full_shape = kspace_dataset.shape[1:]
        
        print(f"Loaded {self.num_slices} slices from {h5_path}")
        print(f"Full k-space shape: {self.full_shape}")
        print(f"Patch size: {patch_size}x{patch_size}")
    
    def __len__(self) -> int:
        return self.num_slices
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.
        
        Returns:
            Dictionary with:
                - 'kspace_under': [2, 64, 64] undersampled k-space
                - 'kspace_full': [2, 64, 64] fully sampled k-space
                - 'mask': [64, 64] undersampling mask
                - 'target': [64, 64] magnitude image from fully sampled
                - 'coords': [64, 64, 2] normalized coordinates
        """
        # Load k-space slice
        with h5py.File(self.h5_path, 'r') as f:
            kspace_slice = f['kspace'][idx]  # [H, W] complex
        
        # Convert to torch tensor
        # fastMRI stores complex as float32 with shape [H, W, 2]
        kspace_tensor = torch.from_numpy(kspace_slice)
        
        # Handle different formats
        if kspace_tensor.dim() == 3 and kspace_tensor.shape[-1] == 2:
            # [H, W, 2] -> [2, H, W]
            kspace_tensor = kspace_tensor.permute(2, 0, 1)
        elif kspace_tensor.dtype == torch.complex64:
            # Complex tensor -> [2, H, W]
            real = kspace_tensor.real
            imag = kspace_tensor.imag
            kspace_tensor = torch.stack([real, imag], dim=0)
        
        # Extract center patch
        kspace_full = extract_center_patch(kspace_tensor, self.patch_size)
        
        # Apply undersampling mask
        # Mask is [H, W], kspace is [2, H, W]
        kspace_under = kspace_full * self.mask.unsqueeze(0)
        
        # Compute target magnitude image from fully sampled
        target = kspace_to_image(kspace_full)
        
        return {
            'kspace_under': kspace_full * self.mask.unsqueeze(0),  # [2, H, W]
            'kspace_full': kspace_full,  # [2, H, W]
            'mask': self.mask,  # [H, W]
            'target': target,  # [H, W]
            'coords': self.coords,  # [H, W, 2]
        }


def visualize_sample(sample: Dict[str, torch.Tensor], save_path: str) -> None:
    """Visualize a sample and save to PNG.
    
    Args:
        sample: Dictionary from ToyFastMRIDataset
        save_path: Path to save visualization
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert tensors to numpy for plotting
    kspace_full_mag = torch.sqrt(sample['kspace_full'][0]**2 + sample['kspace_full'][1]**2).numpy()
    kspace_under_mag = torch.sqrt(sample['kspace_under'][0]**2 + sample['kspace_under'][1]**2).numpy()
    mask = sample['mask'].numpy()
    target = sample['target'].numpy()
    
    # Log scale for k-space
    kspace_full_log = torch.log(torch.tensor(kspace_full_mag) + 1e-10).numpy()
    kspace_under_log = torch.log(torch.tensor(kspace_under_mag) + 1e-10).numpy()
    
    # Plot
    axes[0, 0].imshow(kspace_full_log, cmap='gray')
    axes[0, 0].set_title('Full K-space (log)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(kspace_under_log, cmap='gray')
    axes[0, 1].set_title(f'Undersampled K-space (R={4})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('Sampling Mask')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(target, cmap='gray')
    axes[1, 0].set_title('Target (Fully Sampled)')
    axes[1, 0].axis('off')
    
    # Zero-filled reconstruction from undersampled
    kspace_under_complex = torch.view_as_complex(sample['kspace_under'].permute(1, 2, 0).contiguous())
    zero_filled = torch.abs(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace_under_complex))))
    axes[1, 1].imshow(zero_filled.numpy(), cmap='gray')
    axes[1, 1].set_title('Zero-filled Recon')
    axes[1, 1].axis('off')
    
    # Coordinates visualization (just x coord)
    axes[1, 2].imshow(sample['coords'][:, :, 0].numpy(), cmap='coolwarm')
    axes[1, 2].set_title('X Coordinates (-1 to 1)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ToyFastMRIDataset')
    parser.add_argument('--h5_path', type=str, default=None, help='Path to fastMRI HDF5 file')
    parser.add_argument('--output_dir', type=str, default='demo', help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If no path provided, create synthetic test data
    if args.h5_path is None:
        print("No HDF5 file provided. Creating synthetic test data...")
        
        # Create synthetic k-space data
        h5_path = os.path.join(args.output_dir, 'test_data.h5')
        num_slices = 10
        h, w = 320, 320
        
        # Create phantom-like data
        with h5py.File(h5_path, 'w') as f:
            kspace_data = []
            for i in range(num_slices):
                # Create simple synthetic image (circle)
                y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
                circle = ((x**2 + y**2) < 0.5).float()
                # Add some structure
                image = circle * (1 + 0.5 * torch.sin(10 * x) * torch.cos(10 * y))
                # FFT to k-space
                kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image)))
                # Convert to [H, W, 2] format
                kspace_np = torch.stack([kspace.real, kspace.imag], dim=-1).numpy()
                kspace_data.append(kspace_np)
            
            f.create_dataset('kspace', data=kspace_data)
        
        print(f"Created synthetic test file: {h5_path}")
        args.h5_path = h5_path
    
    # Create dataset
    dataset = ToyFastMRIDataset(
        h5_path=args.h5_path,
        patch_size=64,
        acceleration=4,
        center_fraction=0.08,
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Visualize samples
    num_to_viz = min(args.num_samples, len(dataset))
    for i in range(num_to_viz):
        sample = dataset[i]
        print(f"\nSample {i}:")
        for key, value in sample.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        save_path = os.path.join(args.output_dir, f'sample_{i}.png')
        visualize_sample(sample, save_path)
    
    print(f"\nVisualized {num_to_viz} samples in {args.output_dir}")
