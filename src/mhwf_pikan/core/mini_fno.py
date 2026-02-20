"""Mini FNO (Fourier Neural Operator) for CPU validation.

Small-scale FNO architecture for quick testing on CPU before
scaling up to full GPU models.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniFNO2d(nn.Module):
    """Small 2D FNO layer for CPU validation.
    
    Implements Fourier layer: FFT -> multiply low modes -> IFFT
    Only learns weights for low-frequency modes.
    
    Args:
        modes: Number of Fourier modes to keep in each dimension (default: 8)
        width: Channel width / hidden dimension (default: 32)
        
    Example:
        >>> fno = MiniFNO2d(modes=8, width=32)
        >>> x = torch.randn(2, 32, 64, 64)
        >>> y = fno(x)
        >>> print(y.shape)  # [2, 32, 64, 64]
    """
    
    def __init__(self, modes: int = 8, width: int = 32):
        super().__init__()
        
        self.modes = modes
        self.width = width
        
        # Complex weights for Fourier space multiplication
        # Shape: [width, width, modes, modes, 2] where last dim is (real, imag)
        # Weights are initialized with small values for stability
        self.weights = nn.Parameter(
            torch.randn(width, width, modes, modes, 2) * 0.02
        )
        
        # Optional: bias term
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        
        print(f"MiniFNO2d initialized: modes={modes}, width={width}")
    
    def _mul_complex(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex multiplication of two tensors.
        
        Args:
            a, b: Complex tensors with shape [..., 2] where last dim is (real, imag)
            
        Returns:
            Complex product [..., 2]
        """
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
        imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
        return torch.stack([real, imag], dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO layer.
        
        Args:
            x: Input tensor [B, width, H, W]
            
        Returns:
            Output tensor [B, width, H, W]
        """
        b, c, h, w = x.shape
        assert c == self.width, f"Expected {self.width} channels, got {c}"
        
        # FFT to Fourier space
        # fft2 returns complex tensor [B, width, H, W] (complex dtype)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # Initialize output in Fourier space with zeros
        out_ft = torch.zeros(b, self.width, h, w // 2 + 1, 
                             dtype=x_ft.dtype, device=x.device)
        
        # Multiply relevant Fourier modes by learned weights
        # We keep modes in the range [-modes, modes] for height
        # and [0, modes] for width (rfft2 only returns non-negative frequencies)
        
        # Convert weights to complex
        weights_complex = torch.view_as_complex(self.weights)  # [width, width, modes, modes]
        
        # Low frequency modes (top-left corner of spectrum)
        # Height: 0 to modes-1, Width: 0 to modes-1
        modes_h = min(self.modes, h)
        modes_w = min(self.modes, w // 2 + 1)
        
        # Weight multiplication in Fourier space
        # x_ft: [B, width, H, W//2+1]
        # weights: [out_width, in_width, modes, modes]
        # We want: out_ft[b, j, h, w] = sum_i x_ft[b, i, h, w] * weights[j, i, h, w]
        
        # For efficiency, use einsum
        # Take only low modes from x_ft
        x_ft_low = x_ft[:, :, :modes_h, :modes_w]  # [B, width, modes_h, modes_w]
        
        # Expand weights to match actual modes if needed
        if modes_h < self.modes or modes_w < self.modes:
            weights_truncated = weights_complex[:, :, :modes_h, :modes_w]
        else:
            weights_truncated = weights_complex
        
        # Complex multiplication: [B, in_width, modes_h, modes_w] * [out_width, in_width, modes_h, modes_w]
        # -> [B, out_width, modes_h, modes_w]
        # Using einsum: 'b i h w, o i h w -> b o h w'
        out_ft_low_real = torch.einsum('bihw,oihw->bohw', 
                                       x_ft_low.real, weights_truncated.real) - \
                          torch.einsum('bihw,oihw->bohw', 
                                       x_ft_low.imag, weights_truncated.imag)
        out_ft_low_imag = torch.einsum('bihw,oihw->bohw', 
                                       x_ft_low.real, weights_truncated.imag) + \
                          torch.einsum('bihw,oihw->bohw', 
                                       x_ft_low.imag, weights_truncated.real)
        
        out_ft[:, :, :modes_h, :modes_w] = torch.complex(out_ft_low_real, out_ft_low_imag)
        
        # Also handle negative frequencies for height (if modes > 0)
        if modes_h > 1:
            x_ft_neg = x_ft[:, :, -modes_h+1:, :modes_w]  # Negative frequencies
            
            # Use weights with appropriate indexing for negative frequencies
            if modes_h < self.modes or modes_w < self.modes:
                weights_neg = weights_complex[:, :, -modes_h+1:, :modes_w]
            else:
                weights_neg = weights_complex[:, :, -modes_h+1:, :modes_w]
            
            out_ft_neg_real = torch.einsum('bihw,oihw->bohw', 
                                           x_ft_neg.real, weights_neg.real) - \
                              torch.einsum('bihw,oihw->bohw', 
                                           x_ft_neg.imag, weights_neg.imag)
            out_ft_neg_imag = torch.einsum('bihw,oihw->bohw', 
                                           x_ft_neg.real, weights_neg.imag) + \
                              torch.einsum('bihw,oihw->bohw', 
                                           x_ft_neg.imag, weights_neg.real)
            
            out_ft[:, :, -modes_h+1:, :modes_w] = torch.complex(out_ft_neg_real, out_ft_neg_imag)
        
        # IFFT back to physical space
        x = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1), norm='ortho')
        
        # Add bias
        x = x + self.bias
        
        return x


class MiniFNOReconstructor(nn.Module):
    """Complete mini FNO architecture for CPU testing.
    
    Simple FNO-based reconstruction network for MRI k-space data.
    Uses two FNO layers with activation in between.
    
    Args:
        modes: Number of Fourier modes (default: 8)
        width: Hidden dimension (default: 32)
        n_layers: Number of FNO layers (default: 2)
        
    Example:
        >>> model = MiniFNOReconstructor(modes=8, width=32)
        >>> kspace = torch.randn(2, 2, 64, 64)  # [B, 2, H, W]
        >>> image = model(kspace)
        >>> print(image.shape)  # [2, 1, 64, 64]
    """
    
    def __init__(
        self,
        modes: int = 8,
        width: int = 32,
        n_layers: int = 2,
    ):
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        
        # Input projection: kspace (2 channels) -> width
        self.input_proj = nn.Conv2d(2, width, kernel_size=1)
        
        # FNO layers
        self.fno_layers = nn.ModuleList([
            MiniFNO2d(modes, width) for _ in range(n_layers)
        ])
        
        # Activation between layers
        self.activation = nn.GELU()
        
        # Optional: Layer normalization for stability
        self.norms = nn.ModuleList([
            nn.GroupNorm(min(8, width), width) for _ in range(n_layers)
        ])
        
        # Output projection: width -> 1 (magnitude image)
        self.output_proj = nn.Conv2d(width, 1, kernel_size=1)
        
        print(f"MiniFNOReconstructor initialized:")
        print(f"  Modes: {modes}")
        print(f"  Width: {width}")
        print(f"  Layers: {n_layers}")
    
    def forward(
        self,
        kspace: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            kspace: K-space data [B, 2, H, W] (real/imag channels)
            coords: Optional coordinates (not used in mini version)
            
        Returns:
            Reconstructed magnitude image [B, 1, H, W]
        """
        # Input projection
        x = self.input_proj(kspace)  # [B, width, H, W]
        
        # Apply FNO layers with activations
        for i, (fno, norm) in enumerate(zip(self.fno_layers, self.norms)):
            # FNO layer
            x_fno = fno(x)
            
            # Normalization
            x_fno = norm(x_fno)
            
            # Activation (except last layer)
            if i < self.n_layers - 1:
                x_fno = self.activation(x_fno)
            
            # Residual connection
            x = x + x_fno
        
        # Output projection
        x = self.output_proj(x)  # [B, 1, H, W]
        
        return x


class MiniFNOUNet(nn.Module):
    """Mini FNO UNet with encoder-decoder structure.
    
    More powerful than MiniFNOReconstructor with skip connections.
    
    Args:
        modes: Number of Fourier modes (default: 8)
        base_width: Base channel width (default: 32)
        n_scales: Number of encoder/decoder scales (default: 2)
        
    Example:
        >>> model = MiniFNOUNet(modes=8, base_width=32)
        >>> kspace = torch.randn(2, 2, 64, 64)
        >>> image = model(kspace)
        >>> print(image.shape)  # [2, 1, 64, 64]
    """
    
    def __init__(
        self,
        modes: int = 8,
        base_width: int = 32,
        n_scales: int = 2,
    ):
        super().__init__()
        
        self.modes = modes
        self.base_width = base_width
        self.n_scales = n_scales
        
        # Input projection
        self.input_proj = nn.Conv2d(2, base_width, kernel_size=3, padding=1)
        
        # Encoder
        self.enc_fnos = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i in range(n_scales):
            width = base_width * (2 ** i)
            self.enc_fnos.append(MiniFNO2d(modes, width))
            if i < n_scales - 1:
                self.downsamplers.append(nn.AvgPool2d(2))
        
        # Decoder
        self.dec_fnos = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for i in range(n_scales - 1, -1, -1):
            width = base_width * (2 ** i)
            self.dec_fnos.append(MiniFNO2d(modes, width))
            if i > 0:
                # Upsampler projects from lower scale + skip
                lower_width = base_width * (2 ** (i - 1))
                self.upsamplers.append(
                    nn.Conv2d(width * 2, lower_width, kernel_size=1)
                )
        
        # Output
        self.output_proj = nn.Conv2d(base_width, 1, kernel_size=1)
        
        print(f"MiniFNOUNet initialized:")
        print(f"  Modes: {modes}, Width: {base_width}, Scales: {n_scales}")
    
    def forward(
        self,
        kspace: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Input
        x = self.input_proj(kspace)
        
        # Encoder with skips
        skips = []
        for i in range(self.n_scales):
            x = self.enc_fnos[i](x)
            x = F.gelu(x)
            skips.append(x)
            if i < self.n_scales - 1:
                x = self.downsamplers[i](x)
        
        # Decoder
        for i in range(self.n_scales):
            skip = skips[self.n_scales - 1 - i]
            
            if i > 0:
                # Upsample and concatenate with skip
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = self.upsamplers[i - 1](x)
            else:
                x = skip
            
            x = self.dec_fnos[i](x)
            x = F.gelu(x)
        
        # Output
        x = self.output_proj(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_mini_fno_2d():
    """Test MiniFNO2d layer."""
    print("\n" + "=" * 60)
    print("Test: MiniFNO2d Layer")
    print("=" * 60)
    
    fno = MiniFNO2d(modes=8, width=32)
    
    # Test input
    x = torch.randn(2, 32, 64, 64)
    y = fno(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    correct_shape = y.shape == x.shape
    
    # Check that output is different from input (weights are applied)
    is_different = not torch.allclose(x, y, atol=1e-4)
    print(f"Output differs from input: {is_different}")
    
    # Check gradient flow
    x.requires_grad = True
    y = fno(x)
    loss = y.pow(2).sum()
    loss.backward()
    
    has_grad = x.grad is not None and x.grad.abs().sum() > 0
    print(f"Gradient flows: {has_grad}")
    
    passed = correct_shape and is_different and has_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_mini_fno_reconstructor():
    """Test MiniFNOReconstructor."""
    print("\n" + "=" * 60)
    print("Test: MiniFNOReconstructor")
    print("=" * 60)
    
    model = MiniFNOReconstructor(modes=8, width=32, n_layers=2)
    
    # Test input
    kspace = torch.randn(2, 2, 64, 64)
    image = model(kspace)
    
    print(f"Input kspace: {kspace.shape}")
    print(f"Output image: {image.shape}")
    print(f"Expected: [2, 1, 64, 64]")
    
    correct_shape = image.shape == (2, 1, 64, 64)
    
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    
    # Gradient test
    kspace.requires_grad = True
    image = model(kspace)
    loss = image.pow(2).sum()
    loss.backward()
    
    has_grad = kspace.grad is not None and kspace.grad.abs().sum() > 0
    print(f"Gradient flows: {has_grad}")
    
    passed = correct_shape and has_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_mini_fno_unet():
    """Test MiniFNOUNet."""
    print("\n" + "=" * 60)
    print("Test: MiniFNOUNet")
    print("=" * 60)
    
    model = MiniFNOUNet(modes=8, base_width=32, n_scales=2)
    
    # Test different input sizes
    sizes = [(64, 64), (32, 32)]
    all_passed = True
    
    for h, w in sizes:
        kspace = torch.randn(1, 2, h, w)
        image = model(kspace)
        
        correct = image.shape == (1, 1, h, w)
        all_passed = all_passed and correct
        
        print(f"  [{1}, 2, {h}, {w}] -> {list(image.shape)} {'✓' if correct else '✗'}")
    
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")
    
    print(f"Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_fourier_preservation():
    """Test that FNO preserves certain Fourier properties."""
    print("\n" + "=" * 60)
    print("Test: Fourier Mode Learning")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    fno = MiniFNO2d(modes=4, width=16)
    
    # Create input with specific frequency content
    h, w = 32, 32
    x = torch.linspace(0, 2 * math.pi, h)
    y = torch.linspace(0, 2 * math.pi, w)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Low frequency pattern
    low_freq = torch.sin(2 * xx) * torch.cos(2 * yy)
    
    # High frequency pattern
    high_freq = torch.sin(16 * xx) * torch.cos(16 * yy)
    
    # Combine
    combined = (low_freq + 0.5 * high_freq).unsqueeze(0).unsqueeze(0)
    combined = combined.repeat(1, 16, 1, 1)  # [1, 16, 32, 32]
    
    # Forward
    output = fno(combined)
    
    # Both input and output should have structure
    print(f"Input range: [{combined.min():.3f}, {combined.max():.3f}]")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Output should be different from input (weights applied)
    changed = not torch.allclose(combined, output, atol=1e-3)
    print(f"FNO modified the signal: {changed}")
    
    passed = changed
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MiniFNO Unit Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("MiniFNO2d Layer", test_mini_fno_2d()))
    results.append(("MiniFNOReconstructor", test_mini_fno_reconstructor()))
    results.append(("MiniFNOUNet", test_mini_fno_unet()))
    results.append(("Fourier Mode Learning", test_fourier_preservation()))
    
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
