"""4D Fourier embedding module for spatial and temporal encoding.

Combines spatial Fourier features (x, y coordinates) with sinusoidal
temporal embedding (diffusion timestep) for physics-informed neural networks.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class FourierEmbedding4D(nn.Module):
    """4D Fourier embedding: (x, y) spatial + t diffusion timestep.
    
    Spatial embedding uses Fourier features with frequencies [π, 2π, ..., nπ]
    for each spatial dimension. Temporal embedding uses sinusoidal encoding
    from the Transformers paper.
    
    Args:
        n_spatial_modes: Number of frequency modes for spatial coordinates (default: 16)
        n_time_modes: Number of frequency modes for temporal embedding (default: 64)
        max_time: Maximum timestep value for temporal embedding (default: 1000)
        
    Example:
        >>> emb = FourierEmbedding4D(n_spatial_modes=16, n_time_modes=64)
        >>> coords = torch.randn(2, 64, 64, 2)  # [B, H, W, 2]
        >>> t = torch.tensor([100, 500])  # [B]
        >>> spatial_emb, time_emb = emb(coords, t)
        >>> print(spatial_emb.shape)  # [2, 64, 64, 64]
        >>> print(time_emb.shape)  # [2, 128]
    """
    
    def __init__(
        self,
        n_spatial_modes: int = 16,
        n_time_modes: int = 64,
        max_time: float = 1000.0,
    ):
        super().__init__()
        
        self.n_spatial_modes = n_spatial_modes
        self.n_time_modes = n_time_modes
        self.max_time = max_time
        
        # Spatial frequencies: π, 2π, ..., n*π
        # These create Fourier features at different scales
        spatial_freqs = torch.arange(1, n_spatial_modes + 1, dtype=torch.float32) * math.pi
        self.register_buffer('spatial_freqs', spatial_freqs)  # [n_spatial_modes]
        
        # Time frequencies for sinusoidal embedding
        # Following "Attention Is All You Need" paper
        # freq = 1 / (max_time^(2k / d_model))
        time_freqs = torch.exp(
            torch.arange(0, n_time_modes, dtype=torch.float32) *
            (-math.log(max_time) / n_time_modes)
        )
        self.register_buffer('time_freqs', time_freqs)  # [n_time_modes]
        
        # Calculate output dimensions
        # Spatial: 2 dims (x, y) * 2 funcs (sin, cos) * n_modes
        self.spatial_output_dim = 2 * 2 * n_spatial_modes
        # Temporal: 2 funcs (sin, cos) * n_modes
        self.time_output_dim = 2 * n_time_modes
        
        print(f"FourierEmbedding4D initialized:")
        print(f"  Spatial modes: {n_spatial_modes} -> dim {self.spatial_output_dim}")
        print(f"  Time modes: {n_time_modes} -> dim {self.time_output_dim}")
    
    def embed_spatial(self, coords: torch.Tensor) -> torch.Tensor:
        """Embed 2D spatial coordinates using Fourier features.
        
        For each coordinate dimension and each frequency, computes:
            [sin(freq * x), cos(freq * x)]
        
        Args:
            coords: Normalized coordinates [B, H, W, 2] where last dim is (x, y)
                    Expected range: [-1, 1] or any normalized range
        
        Returns:
            Spatial embedding [B, H, W, 4*n_spatial_modes]
            Layout: [sin(x*f1), cos(x*f1), ..., sin(y*f1), cos(y*f1), ...]
        """
        # coords: [B, H, W, 2]
        b, h, w, _ = coords.shape
        
        # Split into x and y coordinates
        x = coords[..., 0:1]  # [B, H, W, 1]
        y = coords[..., 1:2]  # [B, H, W, 1]
        
        # Expand frequencies for broadcasting
        # spatial_freqs: [n_spatial_modes] -> [1, 1, 1, n_spatial_modes]
        freqs = self.spatial_freqs.view(1, 1, 1, -1)
        
        # Compute x features: [B, H, W, n_spatial_modes]
        x_features = x * freqs  # Broadcasting: [B,H,W,1] * [1,1,1,n] -> [B,H,W,n]
        x_sin = torch.sin(x_features)
        x_cos = torch.cos(x_features)
        
        # Compute y features
        y_features = y * freqs
        y_sin = torch.sin(y_features)
        y_cos = torch.cos(y_features)
        
        # Interleave: [sin(x*f1), cos(x*f1), sin(x*f2), cos(x*f2), ...]
        # Then same for y
        x_emb = torch.stack([x_sin, x_cos], dim=-1)  # [B, H, W, n_modes, 2]
        x_emb = x_emb.view(b, h, w, -1)  # [B, H, W, 2*n_modes]
        
        y_emb = torch.stack([y_sin, y_cos], dim=-1)
        y_emb = y_emb.view(b, h, w, -1)  # [B, H, W, 2*n_modes]
        
        # Concatenate x and y embeddings
        spatial_emb = torch.cat([x_emb, y_emb], dim=-1)  # [B, H, W, 4*n_modes]
        
        return spatial_emb
    
    def embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """Embed diffusion timestep using sinusoidal encoding.
        
        Following the "Attention Is All You Need" paper:
            PE(pos, 2k) = sin(pos / 10000^(2k/d_model))
            PE(pos, 2k+1) = cos(pos / 10000^(2k/d_model))
        
        Args:
            t: Timestep values [B], typically in range [0, max_time]
        
        Returns:
            Time embedding [B, 2*n_time_modes]
        """
        # t: [B]
        b = t.shape[0]
        
        # Expand for broadcasting
        # t: [B, 1], time_freqs: [1, n_time_modes] -> [B, n_time_modes]
        t_expanded = t.unsqueeze(-1)  # [B, 1]
        freqs = self.time_freqs.unsqueeze(0)  # [1, n_time_modes]
        
        # Compute angular frequencies
        angles = t_expanded * freqs  # [B, n_time_modes]
        
        # Apply sin and cos
        emb_sin = torch.sin(angles)  # [B, n_time_modes]
        emb_cos = torch.cos(angles)  # [B, n_time_modes]
        
        # Interleave sin and cos
        time_emb = torch.stack([emb_sin, emb_cos], dim=-1)  # [B, n_time_modes, 2]
        time_emb = time_emb.view(b, -1)  # [B, 2*n_time_modes]
        
        return time_emb
    
    def forward(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing both spatial and temporal embeddings.
        
        Args:
            coords: Spatial coordinates [B, H, W, 2]
            t: Timestep values [B]
        
        Returns:
            Tuple of (spatial_embedding, time_embedding)
            - spatial_embedding: [B, H, W, 4*n_spatial_modes]
            - time_embedding: [B, 2*n_time_modes]
        """
        spatial_emb = self.embed_spatial(coords)
        time_emb = self.embed_time(t)
        
        return spatial_emb, time_emb


class FourierKANLayer(nn.Module):
    """KAN-inspired layer using Fourier basis functions instead of splines.
    
    Combines ideas from Kolmogorov-Arnold Networks with Fourier features
    for efficient function approximation in MRI reconstruction.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        n_fourier_modes: Number of Fourier modes (default: 32)
        
    Example:
        >>> layer = FourierKANLayer(128, 64, n_fourier_modes=32)
        >>> x = torch.randn(2, 100, 128)  # [B, N, in_features]
        >>> y = layer(x)  # [B, N, out_features]
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
        
        # Fourier frequencies
        freqs = torch.arange(1, n_fourier_modes + 1, dtype=torch.float32) * math.pi
        self.register_buffer('freqs', freqs)
        
        # Coefficients for each basis function
        # We have: 1 (constant) + 2 * n_fourier_modes (sin, cos) basis functions
        self.n_basis = 1 + 2 * n_fourier_modes
        
        # Learnable coefficients: [out_features, in_features, n_basis]
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, self.n_basis) * 0.1
        )
        
        # Optional: residual connection scale
        self.residual_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Fourier basis functions.
        
        Args:
            x: Input tensor [..., in_features]
        
        Returns:
            Output tensor [..., out_features]
        """
        # x: [..., in_features]
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)  # [N, in_features]
        n = x_flat.shape[0]
        
        # Expand x for broadcasting with frequencies
        # [N, in_features, 1]
        x_expanded = x_flat.unsqueeze(-1)
        
        # Compute Fourier features
        # freqs: [n_fourier_modes] -> [1, 1, n_fourier_modes]
        freqs_expanded = self.freqs.view(1, 1, -1)
        
        # angles: [N, in_features, n_fourier_modes]
        angles = x_expanded * freqs_expanded
        
        # Basis functions: [N, in_features, n_basis]
        # constant + sin + cos
        basis = torch.cat([
            torch.ones(n, self.in_features, 1, device=x.device),  # constant
            torch.sin(angles),  # sin features
            torch.cos(angles),  # cos features
        ], dim=-1)
        
        # Compute output: sum over input features and basis functions
        # coeffs: [out_features, in_features, n_basis]
        # basis: [N, in_features, n_basis]
        # output: [N, out_features]
        
        # Einstein summation: (o, i, b), (n, i, b) -> (n, o)
        output = torch.einsum('oib,nib->no', self.coeffs, basis)
        
        # Reshape back
        output = output.reshape(*original_shape[:-1], self.out_features)
        
        # Add residual connection
        if self.in_features == self.out_features:
            output = output + self.residual_scale * x
        
        return output


def test_spatial_embedding():
    """Test spatial embedding output shape and properties."""
    print("\n" + "=" * 60)
    print("Test: Spatial Embedding")
    print("=" * 60)
    
    n_spatial_modes = 16
    n_time_modes = 64
    
    emb = FourierEmbedding4D(n_spatial_modes=n_spatial_modes, n_time_modes=n_time_modes)
    
    # Create test coordinates
    b, h, w = 2, 64, 64
    coords = torch.linspace(-1, 1, h)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    
    spatial_emb = emb.embed_spatial(coords)
    
    expected_dim = 4 * n_spatial_modes  # 2 dims * 2 funcs * n_modes
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Output spatial_emb shape: {spatial_emb.shape}")
    print(f"Expected dim: {expected_dim}")
    print(f"Actual dim: {spatial_emb.shape[-1]}")
    
    # Check shape
    correct_shape = spatial_emb.shape == (b, h, w, expected_dim)
    
    # Check that different positions give different embeddings
    emb_00 = spatial_emb[0, 0, 0]
    emb_mid = spatial_emb[0, h//2, w//2]
    emb_end = spatial_emb[0, -1, -1]
    
    different_positions = not torch.allclose(emb_00, emb_mid) and not torch.allclose(emb_mid, emb_end)
    
    print(f"Correct shape: {correct_shape}")
    print(f"Different positions have different embeddings: {different_positions}")
    
    passed = correct_shape and different_positions
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_time_embedding():
    """Test temporal embedding output shape and properties."""
    print("\n" + "=" * 60)
    print("Test: Time Embedding")
    print("=" * 60)
    
    n_spatial_modes = 16
    n_time_modes = 64
    
    emb = FourierEmbedding4D(n_spatial_modes=n_spatial_modes, n_time_modes=n_time_modes)
    
    # Create test timesteps
    b = 4
    t = torch.tensor([0, 250, 500, 1000], dtype=torch.float32)
    
    time_emb = emb.embed_time(t)
    
    expected_dim = 2 * n_time_modes
    
    print(f"Input t shape: {t.shape}")
    print(f"Output time_emb shape: {time_emb.shape}")
    print(f"Expected dim: {expected_dim}")
    
    # Check shape
    correct_shape = time_emb.shape == (b, expected_dim)
    
    # Check that different times give different embeddings
    different_times = not torch.allclose(time_emb[0], time_emb[1])
    
    # Check that t=0 has a specific pattern (sin(0)=0, cos(0)=1)
    # For t=0, all sin terms should be 0, all cos terms should be 1
    emb_0 = time_emb[0]
    sin_vals = emb_0[0::2]  # sin terms at even indices
    cos_vals = emb_0[1::2]  # cos terms at odd indices
    
    sin_near_zero = sin_vals.abs().max() < 1e-5
    cos_near_one = (cos_vals - 1.0).abs().max() < 1e-5
    
    print(f"Correct shape: {correct_shape}")
    print(f"Different times have different embeddings: {different_times}")
    print(f"t=0: sin values near 0: {sin_near_zero}")
    print(f"t=0: cos values near 1: {cos_near_one}")
    
    passed = correct_shape and different_times and sin_near_zero and cos_near_one
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_forward_combined():
    """Test combined forward pass."""
    print("\n" + "=" * 60)
    print("Test: Combined Forward Pass")
    print("=" * 60)
    
    n_spatial_modes = 16
    n_time_modes = 64
    
    emb = FourierEmbedding4D(n_spatial_modes=n_spatial_modes, n_time_modes=n_time_modes)
    
    b, h, w = 2, 32, 32
    coords = torch.randn(b, h, w, 2) * 0.5  # Small random coords
    t = torch.tensor([100.0, 500.0])
    
    spatial_emb, time_emb = emb(coords, t)
    
    expected_spatial = (b, h, w, 4 * n_spatial_modes)
    expected_time = (b, 2 * n_time_modes)
    
    print(f"Input coords: {coords.shape}, t: {t.shape}")
    print(f"Output spatial_emb: {spatial_emb.shape}")
    print(f"Output time_emb: {time_emb.shape}")
    print(f"Expected spatial: {expected_spatial}")
    print(f"Expected time: {expected_time}")
    
    correct_spatial = spatial_emb.shape == expected_spatial
    correct_time = time_emb.shape == expected_time
    
    passed = correct_spatial and correct_time
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_gradient_flow():
    """Test that gradients flow properly."""
    print("\n" + "=" * 60)
    print("Test: Gradient Flow")
    print("=" * 60)
    
    emb = FourierEmbedding4D(n_spatial_modes=8, n_time_modes=16)
    
    coords = torch.randn(2, 16, 16, 2, requires_grad=True)
    t = torch.tensor([100.0, 500.0], requires_grad=True)
    
    spatial_emb, time_emb = emb(coords, t)
    
    # Create a dummy loss
    loss = spatial_emb.pow(2).sum() + time_emb.pow(2).sum()
    loss.backward()
    
    has_coord_grad = coords.grad is not None and coords.grad.abs().sum() > 0
    has_time_grad = t.grad is not None and t.grad.abs().sum() > 0
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Coords gradient exists: {has_coord_grad}")
    print(f"Time gradient exists: {has_time_grad}")
    
    passed = has_coord_grad and has_time_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_fourier_kan_layer():
    """Test Fourier KAN layer."""
    print("\n" + "=" * 60)
    print("Test: Fourier KAN Layer")
    print("=" * 60)
    
    in_features = 32
    out_features = 16
    n_fourier_modes = 16
    
    layer = FourierKANLayer(in_features, out_features, n_fourier_modes)
    
    x = torch.randn(2, 100, in_features)
    y = layer(x)
    
    expected_shape = (2, 100, out_features)
    correct_shape = y.shape == expected_shape
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected shape: {expected_shape}")
    
    # Test gradient flow
    loss = y.pow(2).sum()
    loss.backward()
    
    has_grad = layer.coeffs.grad is not None and layer.coeffs.grad.abs().sum() > 0
    
    print(f"Gradient exists: {has_grad}")
    
    passed = correct_shape and has_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FourierEmbedding4D Unit Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Spatial Embedding", test_spatial_embedding()))
    results.append(("Time Embedding", test_time_embedding()))
    results.append(("Combined Forward", test_forward_combined()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Fourier KAN Layer", test_fourier_kan_layer()))
    
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
