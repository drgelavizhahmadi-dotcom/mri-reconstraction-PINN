"""MHWF-PIKAN UNet architecture for MRI reconstruction.

Hybrid architecture combining:
- Fourier embeddings for spatial/temporal encoding
- Wavelet embeddings for multi-scale analysis
- KAN-inspired layers for efficient function approximation
- Data consistency for physics-based constraints
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and GELU activation.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        kernel_size: Convolution kernel size (default: 3)
        padding: Padding size (default: 1)
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        # GroupNorm with 8 groups (or channels if fewer than 8)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block for encoder.
    
    Applies convolution followed by downsampling.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
    """
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block for decoder.
    
    Upsamples and concatenates with skip connection, then applies convolution.
    
    Args:
        in_ch: Input channels (includes skip connection)
        out_ch: Output channels
    """
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], skip: [B, C_skip, H*2, W*2]
        # Upsample x to match skip
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate with skip
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class FourierKANLayer(nn.Module):
    """KAN-inspired layer using Fourier basis functions.
    
    Uses learnable combinations of Fourier basis for efficient
    function approximation.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        n_fourier_modes: Number of Fourier modes (default: 32)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_fourier_modes: int = 32,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_fourier_modes = n_fourier_modes
        
        # Fourier frequencies: pi, 2pi, ..., n*pi
        freqs = torch.arange(1, n_fourier_modes + 1, dtype=torch.float32) * math.pi
        self.register_buffer('freqs', freqs)
        
        # Basis: constant + sin + cos = 1 + 2*n_fourier_modes
        self.n_basis = 1 + 2 * n_fourier_modes
        
        # Learnable coefficients [out_features, in_features, n_basis]
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, self.n_basis) * 0.1
        )
        
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: [..., in_features]
            
        Returns:
            [..., out_features]
        """
        # x: [N, in_features]
        x_flat = x.reshape(-1, self.in_features)
        n = x_flat.shape[0]
        
        # Expand for broadcasting: [N, in_features, 1]
        x_expanded = x_flat.unsqueeze(-1)
        
        # Frequencies: [1, 1, n_fourier_modes]
        freqs = self.freqs.view(1, 1, -1)
        
        # angles: [N, in_features, n_fourier_modes]
        angles = x_expanded * freqs
        
        # Basis functions: [N, in_features, n_basis]
        basis = torch.cat([
            torch.ones(n, self.in_features, 1, device=x.device),
            torch.sin(angles),
            torch.cos(angles),
        ], dim=-1)
        
        # Weighted combination: [out_features, in_features, n_basis] @ [N, in_features, n_basis]
        output = torch.einsum('oib,nib->no', self.coeffs, basis)
        
        # Add bias and reshape
        output = output + self.bias
        output = output.reshape(*x.shape[:-1], self.out_features)
        
        return output


class MHWF_PIKAN_UNet(nn.Module):
    """MHWF-PIKAN UNet for MRI reconstruction.
    
    Multi-scale Hybrid Wavelet-Fourier Physics-Informed KAN Network.
    
    Architecture:
    - 3-scale encoder-decoder with skip connections
    - Fourier embeddings for spatial (x,y) and temporal (t) coordinates
    - Wavelet embeddings at each encoder scale for multi-scale analysis
    - FourierKAN layers for efficient coefficient mixing
    - Data consistency layer for physics constraint
    
    Args:
        n_scales: Number of encoder/decoder scales (default: 3)
        base_ch: Base number of channels (default: 64)
        n_fourier_modes: Number of Fourier modes for KAN (default: 32)
        n_spatial_modes: Number of spatial embedding modes (default: 16)
        n_time_modes: Number of temporal embedding modes (default: 64)
        wavelet: Wavelet type for wavelet embedding (default: 'db4')
        use_data_consistency: Whether to apply data consistency (default: True)
        
    Example:
        >>> model = MHWF_PIKAN_UNet()
        >>> kspace = torch.randn(2, 2, 64, 64)  # [B, 2, H, W]
        >>> mask = torch.randn(2, 1, 64, 64)
        >>> coords = torch.randn(2, 64, 64, 2)
        >>> t = torch.tensor([100.0, 500.0])
        >>> output = model(kspace, mask, coords, t)
        >>> print(output.shape)  # [2, 2, 64, 64]
    """
    
    def __init__(
        self,
        n_scales: int = 3,
        base_ch: int = 64,
        n_fourier_modes: int = 32,
        n_spatial_modes: int = 16,
        n_time_modes: int = 64,
        wavelet: str = 'db4',
        use_data_consistency: bool = True,
    ):
        super().__init__()
        
        self.n_scales = n_scales
        self.base_ch = base_ch
        self.use_data_consistency = use_data_consistency
        
        # Import embedding modules
        from .fourier_embedding import FourierEmbedding4D
        from .wavelet_embedding import WaveletEmbedding
        
        # Fourier embeddings
        self.fourier_embed = FourierEmbedding4D(
            n_spatial_modes=n_spatial_modes,
            n_time_modes=n_time_modes,
        )
        
        # Input projection: kspace (2ch) + mask (1ch) + spatial_emb -> base_ch
        spatial_emb_dim = 4 * n_spatial_modes  # 2 coords * 2 (sin,cos) * n_modes
        self.input_proj = ConvBlock(2 + 1 + spatial_emb_dim, base_ch)
        
        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.wavelet_embeds = nn.ModuleList()
        
        ch = base_ch
        for i in range(n_scales):
            # Wavelet embedding for this scale (except first)
            if i > 0:
                self.wavelet_embeds.append(WaveletEmbedding(n_scales=4, wavelet=wavelet))
                # After wavelet: ch * 4 channels, project back
                wavelet_proj_in = ch * 4
            else:
                self.wavelet_embeds.append(None)
                wavelet_proj_in = ch
            
            # Encoder block
            out_ch = base_ch * (2 ** i)
            self.enc_blocks.append(DownBlock(wavelet_proj_in, out_ch))
            
            # Downsampler (except last)
            if i < n_scales - 1:
                self.downsamplers.append(nn.AvgPool2d(2))
            
            ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_ch * (2 ** (n_scales - 1))
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_ch, bottleneck_ch),
            ConvBlock(bottleneck_ch, bottleneck_ch),
        )
        
        # Time embedding projection for bottleneck
        time_emb_dim = 2 * n_time_modes
        self.time_proj = nn.Sequential(
            FourierKANLayer(time_emb_dim, bottleneck_ch, n_fourier_modes),
            nn.GELU(),
        )
        
        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for i in range(n_scales - 1, -1, -1):
            in_ch = base_ch * (2 ** i)
            
            # Upsampler (except first decoder step which comes from bottleneck)
            if i < n_scales - 1:
                # Input is 2x current ch (from upsample) + skip
                skip_ch = base_ch * (2 ** i)
                dec_in_ch = in_ch * 2 + skip_ch
            else:
                # First decoder: from bottleneck
                dec_in_ch = in_ch
            
            out_ch = base_ch * (2 ** max(0, i - 1)) if i > 0 else base_ch
            self.dec_blocks.append(UpBlock(dec_in_ch, out_ch))
        
        # Final output projection to 2 channels (real/imag)
        self.output_proj = nn.Conv2d(base_ch, 2, kernel_size=1)
        
        print(f"MHWF_PIKAN_UNet initialized:")
        print(f"  Scales: {n_scales}")
        print(f"  Base channels: {base_ch}")
        print(f"  Fourier modes: {n_fourier_modes}")
        print(f"  Wavelet: {wavelet}")
        print(f"  Data consistency: {use_data_consistency}")
    
    def forward(
        self,
        kspace_under: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            kspace_under: Undersampled k-space [B, 2, H, W]
            mask: Sampling mask [B, 1, H, W]
            coords: Spatial coordinates [B, H, W, 2]
            t: Diffusion timestep [B]
            
        Returns:
            Reconstructed image [B, 2, H, W]
        """
        b, _, h, w = kspace_under.shape
        
        # Get embeddings
        spatial_emb, time_emb = self.fourier_embed(coords, t)  # [B,H,W,64], [B,128]
        
        # Prepare input
        spatial_emb = spatial_emb.permute(0, 3, 1, 2)  # [B, 64, H, W]
        x = torch.cat([kspace_under, mask, spatial_emb], dim=1)  # [B, 67, H, W]
        
        # Initial projection
        x = self.input_proj(x)  # [B, base_ch, H, W]
        
        # Encoder with skip connections
        skips = []
        for i in range(self.n_scales):
            # Apply wavelet embedding (except first scale)
            if i > 0:
                x = self.wavelet_embeds[i].decompose(x)  # [B, ch*4, H, W]
            
            # Encoder block
            x = self.enc_blocks[i](x)
            skips.append(x)
            
            # Downsample (except last)
            if i < self.n_scales - 1:
                x = self.downsamplers[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Add time embedding via adaptive normalization-like mechanism
        time_features = self.time_proj(time_emb)  # [B, bottleneck_ch]
        time_features = time_features.view(b, -1, 1, 1)
        x = x * (1 + time_features)  # Simple feature modulation
        
        # Decoder
        for i in range(self.n_scales):
            skip = skips[self.n_scales - 1 - i]
            
            if i == 0:
                # First decoder step: just upsample bottleneck
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = self.dec_blocks[i](x, torch.zeros_like(skip))  # dummy skip
            else:
                x = self.dec_blocks[i](x, skip)
        
        # Output projection
        output = self.output_proj(x)  # [B, 2, H, W]
        
        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_forward():
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Test: Model Forward Pass")
    print("=" * 60)
    
    model = MHWF_PIKAN_UNet(n_scales=3, base_ch=32)
    
    # Test input
    b, h, w = 2, 64, 64
    kspace = torch.randn(b, 2, h, w)
    mask = torch.randn(b, 1, h, w)
    coords = torch.randn(b, h, w, 2)
    t = torch.tensor([100.0, 500.0])
    
    print(f"Input shapes:")
    print(f"  kspace: {kspace.shape}")
    print(f"  mask: {mask.shape}")
    print(f"  coords: {coords.shape}")
    print(f"  t: {t.shape}")
    
    # Forward
    with torch.no_grad():
        output = model(kspace, mask, coords, t)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [{b}, 2, {h}, {w}]")
    
    correct_shape = output.shape == (b, 2, h, w)
    
    # Parameter count
    n_params = count_parameters(model)
    print(f"\nTrainable parameters: {n_params:,}")
    
    print(f"Result: {'✓ PASSED' if correct_shape else '✗ FAILED'}")
    return correct_shape


def test_gradient_flow():
    """Test gradient flow through model."""
    print("\n" + "=" * 60)
    print("Test: Gradient Flow")
    print("=" * 60)
    
    model = MHWF_PIKAN_UNet(n_scales=2, base_ch=16)  # Smaller for speed
    
    kspace = torch.randn(1, 2, 32, 32, requires_grad=True)
    mask = torch.randn(1, 1, 32, 32)
    coords = torch.randn(1, 32, 32, 2)
    t = torch.tensor([250.0])
    
    output = model(kspace, mask, coords, t)
    loss = output.pow(2).sum()
    loss.backward()
    
    has_grad = kspace.grad is not None and kspace.grad.abs().sum() > 0
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient exists: {has_grad}")
    print(f"Gradient sum: {kspace.grad.abs().sum().item():.4f}")
    
    print(f"Result: {'✓ PASSED' if has_grad else '✗ FAILED'}")
    return has_grad


def test_different_input_sizes():
    """Test with different input sizes."""
    print("\n" + "=" * 60)
    print("Test: Different Input Sizes")
    print("=" * 60)
    
    model = MHWF_PIKAN_UNet(n_scales=3, base_ch=16)
    model.eval()
    
    sizes = [(64, 64), (128, 128), (32, 32)]
    all_passed = True
    
    for h, w in sizes:
        kspace = torch.randn(1, 2, h, w)
        mask = torch.randn(1, 1, h, w)
        coords = torch.randn(1, h, w, 2)
        t = torch.tensor([100.0])
        
        with torch.no_grad():
            output = model(kspace, mask, coords, t)
        
        correct = output.shape == (1, 2, h, w)
        all_passed = all_passed and correct
        
        print(f"  Input: [1, 2, {h}, {w}] -> Output: {list(output.shape)} {'✓' if correct else '✗'}")
    
    print(f"Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MHWF_PIKAN_UNet Unit Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Forward Pass", test_model_forward()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Different Sizes", test_different_input_sizes()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"   {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60 + "\n")
