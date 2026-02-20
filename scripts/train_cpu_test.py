#!/usr/bin/env python
"""CPU training test script for MHWF-PIKAN MRI reconstruction.

Quick training test using:
- ToyFastMRIDataset (64x64 patches, 50 samples)
- MiniFNOReconstructor (small FNO model)
- CPU-friendly settings

Expected runtime: <30 minutes on modern CPU
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mhwf_pikan.core.mini_fno import MiniFNOReconstructor
from mhwf_pikan.data.toy_dataset import ToyFastMRIDataset


def create_synthetic_data(output_path: str, n_samples: int = 50, size: int = 64):
    """Create synthetic fastMRI-like data for testing.
    
    Args:
        output_path: Path to save HDF5 file
        n_samples: Number of slices
        size: Spatial dimensions
    """
    import h5py
    import numpy as np
    
    print(f"Creating synthetic dataset: {n_samples} slices of {size}x{size}")
    
    # Create phantom-like k-space data
    kspace_data = []
    
    for i in range(n_samples):
        # Create simple phantom in image space
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size),
            torch.linspace(-1, 1, size),
            indexing='ij'
        )
        
        # Multiple circles with different intensities
        circle1 = ((x - 0.2)**2 + (y - 0.2)**2 < 0.3).float()
        circle2 = ((x + 0.3)**2 + (y + 0.1)**2 < 0.2).float()
        circle3 = ((x)**2 + (y + 0.4)**2 < 0.15).float()
        
        # Add some texture
        texture = 0.1 * torch.sin(10 * x) * torch.cos(10 * y)
        
        image = (circle1 + 0.7 * circle2 + 0.5 * circle3 + texture).clamp(0, 1)
        
        # Add noise
        image = image + torch.randn_like(image) * 0.05
        image = image.clamp(0, 1)
        
        # FFT to k-space
        kspace_complex = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image)))
        
        # Convert to [H, W, 2] format (real, imag)
        kspace_np = torch.stack([kspace_complex.real, kspace_complex.imag], dim=-1).numpy()
        kspace_data.append(kspace_np)
    
    # Save to HDF5
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('kspace', data=kspace_data)
    
    print(f"Saved synthetic data to {output_path}")
    return output_path


def kspace_to_magnitude_image(kspace: torch.Tensor) -> torch.Tensor:
    """Convert real/imag k-space to magnitude image.
    
    Args:
        kspace: [B, 2, H, W] (real, imag channels)
        
    Returns:
        Magnitude image [B, 1, H, W]
    """
    # Convert to complex
    complex_kspace = torch.view_as_complex(kspace.permute(0, 2, 3, 1).contiguous())
    
    # IFFT2
    image = torch.fft.ifft2(torch.fft.ifftshift(complex_kspace))
    image = torch.fft.fftshift(image)
    
    # Magnitude
    magnitude = torch.abs(image).unsqueeze(1)  # [B, 1, H, W]
    return magnitude


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    print_interval: int = 5,
) -> float:
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        print_interval: Print loss every N batches
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        kspace_full = batch['kspace_full'].to(device)  # [B, 2, H, W]
        kspace_under = batch['kspace_under'].to(device)  # [B, 2, H, W]
        mask = batch['mask'].to(device)  # [B, H, W]
        target = batch['target'].to(device)  # [B, H, W]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(kspace_under)  # [B, 1, H, W]
        
        # Compute loss
        # Target is [B, H, W], pred is [B, 1, H, W]
        target_expanded = target.unsqueeze(1)  # [B, 1, H, W]
        loss = criterion(pred, target_expanded)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Print every N batches
        if (batch_idx + 1) % print_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.6f}")
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            kspace_under = batch['kspace_under'].to(device)
            target = batch['target'].to(device)
            
            pred = model(kspace_under)
            target_expanded = target.unsqueeze(1)
            loss = criterion(pred, target_expanded)
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='CPU Training Test for MHWF-PIKAN')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to fastMRI HDF5 file (auto-generates if not provided)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--modes', type=int, default=8,
                        help='FNO modes (default: 8)')
    parser.add_argument('--width', type=int, default=32,
                        help='FNO width (default: 32)')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints (default: checkpoints)')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of synthetic samples to generate (default: 50)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cpu')
    print(f"\n{'=' * 60}")
    print("MHWF-PIKAN CPU Training Test")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"FNO modes: {args.modes}, width: {args.width}")
    print(f"{'=' * 60}\n")
    
    # Create or load data
    if args.data_path is None:
        data_path = 'data/toy_mri.h5'
        args.data_path = create_synthetic_data(data_path, n_samples=args.n_samples)
    
    # Create dataset
    print("Loading dataset...")
    dataset = ToyFastMRIDataset(
        h5_path=args.data_path,
        patch_size=64,
        acceleration=4,
        center_fraction=0.08,
    )
    
    # Split into train/val (80/20)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for CPU to avoid multiprocessing overhead
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Create model
    print("\nInitializing model...")
    model = MiniFNOReconstructor(
        modes=args.modes,
        width=args.width,
        n_layers=2,
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\n{'=' * 60}")
    print("Starting Training")
    print(f"{'=' * 60}\n")
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            print_interval=5
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Timing
        epoch_time = time.time() - epoch_start
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"* New best validation loss!")
    
    # Total training time
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"{'=' * 60}")
    
    # Save final checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.output_dir,
        f'mini_fno_epoch{args.epochs}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
    )
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args),
    }, checkpoint_path)
    
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
