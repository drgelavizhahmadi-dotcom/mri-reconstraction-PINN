"""Full FNO-KAN Reconstructor for GPU training.

Combines Fourier Neural Operators with Kolmogorov-Arnold Networks
for efficient and expressive MRI reconstruction.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def b_spline_basis(x: torch.Tensor, grid: torch.Tensor, spline_order: int) -> torch.Tensor:
    """Compute B-spline basis functions.
    
    Args:
        x: Input values [..., in_features]
        grid: Knot grid [in_features, grid_size + 2 * spline_order + 1]
        spline_order: Order of B-spline (3 = cubic)
        
    Returns:
        B-spline basis [..., in_features, grid_size + spline_order]
    """
    # x: [..., in_features]
    # Expand for broadcasting with grid
    x_expanded = x.unsqueeze(-1)  # [..., in_features, 1]
    
    # Initialize B-spline of order 0
    # grid: [in_features, n_knots]
    # Check if x is within each interval
    basis = ((x_expanded >= grid[..., :-1]) & (x_expanded < grid[..., 1:])).float()
    
    # Recurrence for higher orders
    for k in range(1, spline_order + 1):
        # Left part
        left_num = x_expanded - grid[..., :-(k + 1)]
        left_den = grid[..., k:-1] - grid[..., :-(k + 1)] + 1e-8
        left = left_num / left_den
        
        # Right part  
        right_num = grid[..., k + 1:] - x_expanded
        right_den = grid[..., k + 1:] - grid[..., 1:-k] + 1e-8
        right = right_num / right_den
        
        basis = left * basis[..., :-1] + right * basis[..., 1:]
    
    return basis


class KANLinear(nn.Module):
    """KAN layer with B-spline activations.
    
    Simplified from pykan for efficiency. Combines:
    - Base activation: linear + SiLU
    - Spline activation: B-spline basis * learnable weights
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        grid_size: Number of grid intervals (default: 5)
        spline_order: B-spline order (default: 3, cubic)
        
    Example:
        >>> kan = KANLinear(64, 64, grid_size=5, spline_order=3)
        >>> x = torch.randn(2, 100, 64)
        >>> y = kan(x)
        >>> print(y.shape)  # [2, 100, 64]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid for B-splines: [in_features, n_knots]
        # Extended grid for boundary handling
        n_knots = grid_size + 2 * spline_order + 1
        
        # Initialize uniform grid from -1 to 1
        grid_range = [-1, 1]
        grid = torch.linspace(grid_range[0], grid_range[1], n_knots).unsqueeze(0).repeat(in_features, 1)
        self.register_buffer('grid', grid)
        
        # Base weights for linear + SiLU path
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        # Spline weights: [out_features, in_features, grid_size + spline_order]
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, n_basis) * 0.1)
        self.spline_bias = nn.Parameter(torch.zeros(out_features))
        
        # Scale factor for spline contribution
        self.spline_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input [..., in_features]
            
        Returns:
            Output [..., out_features]
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # [N, in_features]
        
        # Base path: linear + SiLU
        base_output = F.linear(F.silu(x_flat), self.base_weight, self.base_bias)
        
        # Spline path: B-spline basis * weights
        # Normalize x to grid range [-1, 1] if needed
        x_normalized = torch.tanh(x_flat)  # Ensure in [-1, 1]
        
        # Compute B-spline basis
        basis = b_spline_basis(x_normalized, self.grid, self.spline_order)
        # basis: [N, in_features, n_basis]
        
        # Weighted combination: [out_features, in_features, n_basis] @ [N, in_features, n_basis]
        spline_output = torch.einsum('oib,nib->no', self.spline_weight, basis)
        spline_output = spline_output + self.spline_bias
        
        # Combine base and spline
        output = base_output + self.spline_scale * spline_output
        
        # Reshape back
        output = output.reshape(*original_shape[:-1], self.out_features)
        
        return output


class SpatialKAN(nn.Module):
    """KAN operating on spatial features [B, C, H, W].
    
    Applies KAN across the channel dimension at each spatial location.
    
    Args:
        channels: Number of input/output channels
        grid_size: B-spline grid size (default: 5)
        spline_order: B-spline order (default: 3)
        
    Example:
        >>> skan = SpatialKAN(64, grid_size=5)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> y = skan(x)
        >>> print(y.shape)  # [2, 64, 32, 32]
    """
    
    def __init__(
        self,
        channels: int,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        
        self.channels = channels
        self.kan = KANLinear(channels, channels, grid_size, spline_order)
        
        # Layer normalization for stability
        self.norm = nn.GroupNorm(min(8, channels), channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            Output [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Reshape to [B*H*W, C]
        x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_flat = x_perm.reshape(-1, c)  # [B*H*W, C]
        
        # Apply KAN
        y_flat = self.kan(x_flat)  # [B*H*W, C]
        
        # Reshape back
        y = y_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply normalization
        y = self.norm(y)
        
        # Residual connection
        return x + y


class FNOBlock2d(nn.Module):
    """FNO block with spectral convolution and skip connection.
    
    Args:
        modes: Number of Fourier modes
        width: Channel width
        
    Example:
        >>> fno = FNOBlock2d(modes=16, width=64)
        >>> x = torch.randn(2, 64, 64, 64)
        >>> y = fno(x)
        >>> print(y.shape)  # [2, 64, 64, 64]
    """
    
    def __init__(self, modes: int, width: int):
        super().__init__()
        
        self.modes = modes
        self.width = width
        
        # Complex weights for spectral convolution
        self.weights = nn.Parameter(torch.randn(width, width, modes, modes, 2) * 0.02)
        
        # Spatial convolution path (1x1 conv)
        self.spatial_conv = nn.Conv2d(width, width, kernel_size=1)
        
        # Layer norm
        self.norm = nn.GroupNorm(min(8, width), width)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        b, c, h, w = x.shape
        
        # Spectral path
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')  # [B, C, H, W//2+1]
        
        out_ft = torch.zeros(b, self.width, h, w // 2 + 1, 
                            dtype=x_ft.dtype, device=x.device)
        
        # Multiply low modes
        modes_h = min(self.modes, h)
        modes_w = min(self.modes, w // 2 + 1)
        
        weights_complex = torch.view_as_complex(self.weights[:, :, :modes_h, :modes_w])
        
        x_ft_low = x_ft[:, :, :modes_h, :modes_w]
        
        # Einstein summation for complex multiplication
        out_ft_real = torch.einsum('bchw,ochw->bohw', x_ft_low.real, weights_complex.real) - \
                      torch.einsum('bchw,ochw->bohw', x_ft_low.imag, weights_complex.imag)
        out_ft_imag = torch.einsum('bchw,ochw->bohw', x_ft_low.real, weights_complex.imag) + \
                      torch.einsum('bchw,ochw->bohw', x_ft_low.imag, weights_complex.real)
        
        out_ft[:, :, :modes_h, :modes_w] = torch.complex(out_ft_real, out_ft_imag)
        
        # Negative frequencies for height
        if modes_h > 1:
            x_ft_neg = x_ft[:, :, -modes_h+1:, :modes_w]
            weights_neg = torch.view_as_complex(self.weights[:, :, -modes_h+1:, :modes_w])
            
            out_neg_real = torch.einsum('bchw,ochw->bohw', x_ft_neg.real, weights_neg.real) - \
                           torch.einsum('bchw,ochw->bohw', x_ft_neg.imag, weights_neg.imag)
            out_neg_imag = torch.einsum('bchw,ochw->bohw', x_ft_neg.real, weights_neg.imag) + \
                           torch.einsum('bchw,ochw->bohw', x_ft_neg.imag, weights_neg.real)
            
            out_ft[:, :, -modes_h+1:, :modes_w] = torch.complex(out_neg_real, out_neg_imag)
        
        x_spectral = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1), norm='ortho')
        
        # Spatial path
        x_spatial = self.spatial_conv(x)
        
        # Combine and normalize
        y = x_spectral + x_spatial
        y = self.norm(y)
        y = F.gelu(y)
        
        return y


class FNOKANReconstructor(nn.Module):
    """Full FNO-KAN architecture for GPU training.
    
    Combines Fourier Neural Operators for global context with
    KAN layers for local refinement.
    
    Args:
        modes: Number of Fourier modes (default: 16)
        width: Base channel width (default: 64)
        n_fno_layers: Number of FNO layers (default: 4)
        n_kan_layers: Number of KAN refinement layers (default: 2)
        grid_size: B-spline grid size for KAN (default: 5)
        spline_order: B-spline order (default: 3)
        
    Example:
        >>> model = FNOKANReconstructor(modes=16, width=64)
        >>> kspace = torch.randn(2, 2, 64, 64)
        >>> coords = torch.randn(2, 64, 64, 2)
        >>> t = torch.tensor([100.0, 500.0])
        >>> image = model(kspace, coords, t)
        >>> print(image.shape)  # [2, 1, 64, 64]
    """
    
    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_fno_layers: int = 4,
        n_kan_layers: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_fno_layers = n_fno_layers
        self.n_kan_layers = n_kan_layers
        
        # Import Fourier embedding
        from .fourier_embedding import FourierEmbedding4D
        
        # Fourier embedding for spatial coordinates and time
        self.fourier_embed = FourierEmbedding4D(
            n_spatial_modes=16,
            n_time_modes=64,
        )
        
        # Input projection: kspace(2) + spatial_emb(64) -> width
        self.input_proj = nn.Sequential(
            nn.Conv2d(2 + 64, width, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, width), width),
            nn.GELU(),
        )
        
        # FNO layers with skip connections
        self.fno_layers = nn.ModuleList([
            FNOBlock2d(modes, width) for _ in range(n_fno_layers)
        ])
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(128, width),  # 2 * n_time_modes = 128
            nn.GELU(),
        )
        
        # KAN refinement layers
        self.kan_layers = nn.ModuleList([
            SpatialKAN(width, grid_size, spline_order) 
            for _ in range(n_kan_layers)
        ])
        
        # Output projection: width -> 1 (magnitude image)
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, width // 2), width // 2),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, kernel_size=1),
        )
        
        print(f"FNOKANReconstructor initialized:")
        print(f"  Modes: {modes}, Width: {width}")
        print(f"  FNO layers: {n_fno_layers}, KAN layers: {n_kan_layers}")
        print(f"  Grid size: {grid_size}, Spline order: {spline_order}")
        
    def forward(
        self,
        kspace: torch.Tensor,
        coords: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            kspace: K-space data [B, 2, H, W]
            coords: Spatial coordinates [B, H, W, 2]
            t: Diffusion timestep [B] (optional)
            
        Returns:
            Reconstructed image [B, 1, H, W]
        """
        b, _, h, w = kspace.shape
        
        # Handle missing time input
        if t is None:
            t = torch.zeros(b, device=kspace.device)
        
        # Get Fourier embeddings
        spatial_emb, time_emb = self.fourier_embed(coords, t)
        
        # Prepare input
        spatial_emb = spatial_emb.permute(0, 3, 1, 2)  # [B, 64, H, W]
        x = torch.cat([kspace, spatial_emb], dim=1)  # [B, 66, H, W]
        
        # Initial projection
        x = self.input_proj(x)  # [B, width, H, W]
        
        # Apply time embedding modulation
        time_features = self.time_proj(time_emb)  # [B, width]
        time_features = time_features.view(b, -1, 1, 1)
        x = x * (1 + time_features)  # Feature modulation
        
        # FNO layers with residual connections
        for fno in self.fno_layers:
            x_new = fno(x)
            x = x + x_new  # Residual
        
        # KAN refinement layers
        for kan in self.kan_layers:
            x = kan(x)
        
        # Output projection
        image = self.output_proj(x)  # [B, 1, H, W]
        
        return image


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_kan_linear():
    """Test KANLinear layer."""
    print("\n" + "=" * 60)
    print("Test: KANLinear Layer")
    print("=" * 60)
    
    kan = KANLinear(64, 64, grid_size=5, spline_order=3)
    
    x = torch.randn(2, 100, 64)
    y = kan(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    correct_shape = y.shape == x.shape
    
    # Gradient test
    x.requires_grad = True
    y = kan(x)
    loss = y.pow(2).sum()
    loss.backward()
    
    has_grad = x.grad is not None and kan.spline_weight.grad is not None
    
    print(f"Shape correct: {correct_shape}")
    print(f"Gradient flows: {has_grad}")
    
    passed = correct_shape and has_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_spatial_kan():
    """Test SpatialKAN."""
    print("\n" + "=" * 60)
    print("Test: SpatialKAN")
    print("=" * 60)
    
    skan = SpatialKAN(64, grid_size=5, spline_order=3)
    
    x = torch.randn(2, 64, 32, 32)
    y = skan(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    correct_shape = y.shape == x.shape
    
    # Output should be different (residual + transformation)
    is_different = not torch.allclose(x, y, atol=1e-4)
    
    print(f"Shape correct: {correct_shape}")
    print(f"Output transformed: {is_different}")
    
    passed = correct_shape and is_different
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_fno_block():
    """Test FNOBlock2d."""
    print("\n" + "=" * 60)
    print("Test: FNOBlock2d")
    print("=" * 60)
    
    fno = FNOBlock2d(modes=8, width=32)
    
    x = torch.randn(2, 32, 64, 64)
    y = fno(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    correct_shape = y.shape == x.shape
    
    # Gradient test
    x.requires_grad = True
    y = fno(x)
    loss = y.pow(2).sum()
    loss.backward()
    
    has_grad = x.grad is not None and fno.weights.grad is not None
    
    print(f"Shape correct: {correct_shape}")
    print(f"Gradient flows: {has_grad}")
    
    passed = correct_shape and has_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_fnokan_reconstructor():
    """Test full FNOKANReconstructor."""
    print("\n" + "=" * 60)
    print("Test: FNOKANReconstructor")
    print("=" * 60)
    
    model = FNOKANReconstructor(
        modes=8,
        width=32,
        n_fno_layers=2,
        n_kan_layers=1,
        grid_size=3,
    )
    
    # Test input
    b, h, w = 2, 64, 64
    kspace = torch.randn(b, 2, h, w)
    coords = torch.randn(b, h, w, 2)
    t = torch.tensor([100.0, 500.0])
    
    print(f"Input shapes:")
    print(f"  kspace: {kspace.shape}")
    print(f"  coords: {coords.shape}")
    print(f"  t: {t.shape}")
    
    # Forward
    image = model(kspace, coords, t)
    
    print(f"\nOutput image: {image.shape}")
    print(f"Expected: [{b}, 1, {h}, {w}]")
    
    correct_shape = image.shape == (b, 1, h, w)
    
    n_params = count_parameters(model)
    print(f"\nTrainable parameters: {n_params:,}")
    
    # Gradient test
    kspace.requires_grad = True
    image = model(kspace, coords, t)
    loss = image.pow(2).sum()
    loss.backward()
    
    has_grad = kspace.grad is not None and kspace.grad.abs().sum() > 0
    print(f"Gradient flows: {has_grad}")
    
    passed = correct_shape and has_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_without_time():
    """Test without time input."""
    print("\n" + "=" * 60)
    print("Test: Without Time Input")
    print("=" * 60)
    
    model = FNOKANReconstructor(modes=4, width=16, n_fno_layers=1, n_kan_layers=1)
    
    kspace = torch.randn(1, 2, 32, 32)
    coords = torch.randn(1, 32, 32, 2)
    
    image = model(kspace, coords, t=None)
    
    print(f"Input: {kspace.shape}")
    print(f"Output: {image.shape}")
    
    passed = image.shape == (1, 1, 32, 32)
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FNOKAN Unit Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("KANLinear", test_kan_linear()))
    results.append(("SpatialKAN", test_spatial_kan()))
    results.append(("FNOBlock2d", test_fno_block()))
    results.append(("FNOKANReconstructor", test_fnokan_reconstructor()))
    results.append(("Without Time", test_without_time()))
    
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
