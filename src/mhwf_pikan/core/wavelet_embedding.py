"""Multi-scale wavelet embedding module for k-space/image decomposition.

Uses 2D discrete wavelet transform to extract multi-scale features
that capture both frequency and spatial localization information.
"""

from typing import List, Optional, Tuple

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletEmbedding(nn.Module):
    """Multi-scale wavelet decomposition for images/k-space.
    
    Decomposes input into wavelet coefficients at multiple scales and
    upsamples them to the original resolution for multi-scale feature
    representation.
    
    Args:
        n_scales: Number of wavelet decomposition levels (default: 4)
        wavelet: Wavelet type, e.g., 'db4', 'haar', 'sym4' (default: 'db4')
        mode: Signal extension mode for DWT (default: 'symmetric')
        
    Example:
        >>> wvt = WaveletEmbedding(n_scales=4, wavelet='db4')
        >>> image = torch.randn(2, 1, 64, 64)
        >>> features = wvt.decompose(image)
        >>> print(features.shape)  # [2, 4, 64, 64]
        
        >>> # With adaptive gating using time embedding
        >>> t_emb = torch.randn(2, 64)
        >>> features = wvt(image, t_emb)
    """
    
    def __init__(
        self,
        n_scales: int = 4,
        wavelet: str = 'db4',
        mode: str = 'symmetric',
    ):
        super().__init__()
        
        self.n_scales = n_scales
        self.wavelet = wavelet
        self.mode = mode
        
        # Verify wavelet exists
        try:
            wt = pywt.Wavelet(wavelet)
            print(f"WaveletEmbedding initialized:")
            print(f"  Wavelet: {wavelet} ({wt.family_name})")
            print(f"  Decomposition levels: {n_scales}")
            print(f"  Filter length: {wt.dec_len}")
        except ValueError as e:
            raise ValueError(f"Invalid wavelet '{wavelet}': {e}")
        
        # Output channels = n_scales (one per scale level)
        # Each scale level combines detail coefficients
        self.output_channels_per_input = n_scales
    
    def _wavelet_dec_2d(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Apply 2D wavelet decomposition to a single channel.
        
        Args:
            x: Single channel tensor [H, W]
            
        Returns:
            Tuple of (approximation_coeffs, list_of_detail_coeffs)
            where list_of_detail_coeffs contains tuples of (cH, cV, cD) for each level
        """
        # Convert to numpy for pywt
        x_np = x.detach().cpu().numpy()
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(x_np, self.wavelet, mode=self.mode, level=self.n_scales)
        
        # coeffs[0] is approximation (cA)
        # coeffs[1:] are detail coefficients (cH, cV, cD) for each level
        cA = coeffs[0]
        details = coeffs[1:]
        
        # Convert back to torch
        cA_tensor = torch.from_numpy(cA).to(x.device, x.dtype)
        detail_tensors = [
            (torch.from_numpy(cH).to(x.device, x.dtype),
             torch.from_numpy(cV).to(x.device, x.dtype),
             torch.from_numpy(cD).to(x.device, x.dtype))
            for cH, cV, cD in details
        ]
        
        return cA_tensor, detail_tensors
    
    def _upsample_to_size(
        self,
        x: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Upsample tensor to target size using interpolation.
        
        Args:
            x: Input tensor [H', W']
            target_size: (H, W) target size
            
        Returns:
            Upsampled tensor [H, W]
        """
        # Add batch and channel dims for interpolate: [1, 1, H', W']
        x_expanded = x.unsqueeze(0).unsqueeze(0)
        
        # Upsample
        x_upsampled = F.interpolate(
            x_expanded,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )
        
        # Remove batch and channel dims: [H, W]
        return x_upsampled.squeeze(0).squeeze(0)
    
    def _combine_detail_coeffs(
        self,
        cH: torch.Tensor,
        cV: torch.Tensor,
        cD: torch.Tensor,
    ) -> torch.Tensor:
        """Combine horizontal, vertical, diagonal detail coefficients.
        
        Combines using magnitude: sqrt(cH^2 + cV^2 + cD^2)
        
        Args:
            cH, cV, cD: Detail coefficients [H', W']
            
        Returns:
            Combined magnitude [H', W']
        """
        # Magnitude of gradient-like information
        magnitude = torch.sqrt(cH**2 + cV**2 + cD**2 + 1e-8)
        return magnitude
    
    def decompose(self, image: torch.Tensor) -> torch.Tensor:
        """Decompose image into multi-scale wavelet features.
        
        Performs n_scales-level wavelet decomposition and upsamples
        all coefficients to the original resolution.
        
        Args:
            image: Input image [B, C, H, W]
            
        Returns:
            Wavelet features [B, n_scales*C, H, W]
            Each group of n_scales channels corresponds to one input channel,
            ordered from coarsest to finest scale.
        """
        b, c, h, w = image.shape
        
        # Output will have n_scales channels per input channel
        output_channels = c * self.n_scales
        output = torch.zeros(b, output_channels, h, w, device=image.device, dtype=image.dtype)
        
        for batch_idx in range(b):
            for ch_idx in range(c):
                # Extract single channel
                x = image[batch_idx, ch_idx]  # [H, W]
                
                # Wavelet decomposition
                cA, details = self._wavelet_dec_2d(x)
                
                # Upsample approximation (coarsest scale)
                cA_up = self._upsample_to_size(cA, (h, w))
                
                # Process each detail level
                scale_features = [cA_up]  # Start with approximation
                
                for level, (cH, cV, cD) in enumerate(details):
                    # Combine detail coefficients
                    detail_combined = self._combine_detail_coeffs(cH, cV, cD)
                    
                    # Upsample to original size
                    detail_up = self._upsample_to_size(detail_combined, (h, w))
                    scale_features.append(detail_up)
                
                # We have n_scales + 1 features (approx + n_scales details)
                # But we want exactly n_scales output channels
                # Combine approximation with first detail level or skip approximation
                # Strategy: use detail levels only (finer scales)
                # Or: average approximation with coarsest detail
                
                # Let's use: [approx+coarse, fine_1, fine_2, ..., fine_{n-1}]
                # This gives n_scales channels
                if len(scale_features) == self.n_scales + 1:
                    # Combine two coarsest levels
                    coarse_combined = (scale_features[0] + scale_features[1]) / 2
                    selected_features = [coarse_combined] + scale_features[2:]
                else:
                    selected_features = scale_features[:self.n_scales]
                
                # Stack features: [n_scales, H, W]
                features = torch.stack(selected_features, dim=0)
                
                # Place in output
                out_start = ch_idx * self.n_scales
                out_end = out_start + self.n_scales
                output[batch_idx, out_start:out_end] = features
        
        return output
    
    def forward(
        self,
        image: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional adaptive gating.
        
        Args:
            image: Input image [B, C, H, W]
            t_emb: Optional time embedding [B, D] for adaptive gating
            
        Returns:
            Wavelet features [B, n_scales*C, H, W]
            If t_emb provided, features are adaptively gated by time embedding.
        """
        # Get wavelet features
        wavelet_features = self.decompose(image)  # [B, n_scales*C, H, W]
        
        if t_emb is None:
            return wavelet_features
        
        # Adaptive gating using time embedding
        b, n_features, h, w = wavelet_features.shape
        n_scales = self.n_scales
        
        # Time embedding projection to scale weights
        # t_emb: [B, D] -> scale_weights: [B, n_scales]
        t_emb_dim = t_emb.shape[-1]
        
        # Simple linear projection
        scale_weights = torch.matmul(t_emb, torch.randn(t_emb_dim, n_scales, device=t_emb.device))
        scale_weights = torch.sigmoid(scale_weights)  # [B, n_scales]
        
        # Reshape for broadcasting: [B, n_scales, 1, 1]
        scale_weights = scale_weights.view(b, n_scales, 1, 1)
        
        # Apply gating per input channel group
        num_input_channels = n_features // n_scales
        gated_features = []
        
        for ch in range(num_input_channels):
            start = ch * n_scales
            end = start + n_scales
            features_ch = wavelet_features[:, start:end]  # [B, n_scales, H, W]
            
            # Gate by scale weights
            gated_ch = features_ch * scale_weights
            gated_features.append(gated_ch)
        
        # Concatenate back
        output = torch.cat(gated_features, dim=1)  # [B, n_scales*C, H, W]
        
        return output


class WaveletKANLayer(nn.Module):
    """KAN-inspired layer using wavelet basis functions.
    
    Uses learnable combinations of wavelet functions as basis
    for efficient function approximation.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        n_wavelets: Number of wavelet basis functions (default: 16)
        wavelet: Wavelet type (default: 'db4')
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_wavelets: int = 16,
        wavelet: str = 'db4',
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_wavelets = n_wavelets
        
        # Get wavelet filters
        wt = pywt.Wavelet(wavelet)
        
        # Learnable wavelet coefficient combinations
        # For each output, we combine wavelet bases across inputs
        self.wavelet_weights = nn.Parameter(
            torch.randn(out_features, in_features, n_wavelets) * 0.1
        )
        
        # Scale parameters for wavelet dilation
        self.scales = nn.Parameter(
            torch.exp(torch.linspace(0, 2, n_wavelets))  # Increasing scales
        )
        
        # Translation parameters
        self.translations = nn.Parameter(
            torch.zeros(n_wavelets)
        )
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def _wavelet_basis(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate wavelet basis functions at x.
        
        Uses Morlet-like wavelet: cos(5*t) * exp(-t^2/2)
        
        Args:
            x: Input values [..., in_features]
            
        Returns:
            Wavelet basis values [..., in_features, n_wavelets]
        """
        # x: [..., in_features] -> [..., in_features, 1]
        x_expanded = x.unsqueeze(-1)
        
        # Apply dilation and translation: (x - b) / a
        scales = self.scales.view(1, 1, -1)  # [1, 1, n_wavelets]
        translations = self.translations.view(1, 1, -1)
        
        t = (x_expanded - translations) / (scales + 1e-6)
        
        # Morlet wavelet: cos(5*t) * exp(-t^2/2)
        wavelet = torch.cos(5 * t) * torch.exp(-t**2 / 2)
        
        return wavelet
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using wavelet basis functions.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # Evaluate wavelet basis
        basis = self._wavelet_basis(x)  # [..., in_features, n_wavelets]
        
        # Weighted combination
        # wavelet_weights: [out_features, in_features, n_wavelets]
        # basis: [..., in_features, n_wavelets]
        # output: [..., out_features]
        
        output = torch.einsum('oik,...ik->...o', self.wavelet_weights, basis)
        
        # Add bias
        output = output + self.bias
        
        return output


def test_decomposition_shape():
    """Test that decomposition produces correct output shape."""
    print("\n" + "=" * 60)
    print("Test: Decomposition Shape")
    print("=" * 60)
    
    wvt = WaveletEmbedding(n_scales=4, wavelet='db4')
    
    # Test with different input shapes
    test_cases = [
        (2, 1, 64, 64),
        (1, 2, 128, 128),
        (4, 1, 32, 32),
    ]
    
    all_passed = True
    for shape in test_cases:
        b, c, h, w = shape
        image = torch.randn(shape)
        
        output = wvt.decompose(image)
        expected_shape = (b, c * 4, h, w)  # 4 scales per channel
        
        shape_correct = output.shape == expected_shape
        all_passed = all_passed and shape_correct
        
        print(f"Input: {shape} -> Output: {output.shape}, Expected: {expected_shape}, {'✓' if shape_correct else '✗'}")
    
    print(f"Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_multi_scale_content():
    """Test that different scales capture different frequency content."""
    print("\n" + "=" * 60)
    print("Test: Multi-scale Content")
    print("=" * 60)
    
    wvt = WaveletEmbedding(n_scales=3, wavelet='haar')
    
    # Create an image with high frequency content
    x = torch.linspace(0, 8 * 3.14159, 64)
    y = torch.linspace(0, 8 * 3.14159, 64)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # High frequency pattern
    high_freq_image = torch.sin(16 * xx) * torch.cos(16 * yy)
    image = high_freq_image.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]
    
    # Decompose
    features = wvt.decompose(image)  # [1, 3, 64, 64]
    
    # Check that different scales have different magnitudes
    scale_0 = features[0, 0].abs().mean().item()
    scale_1 = features[0, 1].abs().mean().item()
    scale_2 = features[0, 2].abs().mean().item()
    
    print(f"Scale 0 (coarse) mean abs: {scale_0:.4f}")
    print(f"Scale 1 (medium) mean abs: {scale_1:.4f}")
    print(f"Scale 2 (fine) mean abs: {scale_2:.4f}")
    
    # For high frequency content, finer scales should have higher energy
    different_content = not (abs(scale_0 - scale_1) < 0.01 and abs(scale_1 - scale_2) < 0.01)
    
    print(f"Different scales have different content: {different_content}")
    print(f"Result: {'✓ PASSED' if different_content else '✗ FAILED'}")
    
    return different_content


def test_gradient_flow():
    """Test that gradients flow through the module."""
    print("\n" + "=" * 60)
    print("Test: Gradient Flow")
    print("=" * 60)
    
    wvt = WaveletEmbedding(n_scales=4, wavelet='db4')
    
    image = torch.randn(2, 1, 64, 64, requires_grad=True)
    
    output = wvt.decompose(image)
    loss = output.pow(2).sum()
    loss.backward()
    
    has_grad = image.grad is not None and image.grad.abs().sum() > 0
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Input gradient exists: {has_grad}")
    print(f"Gradient sum: {image.grad.abs().sum().item():.4f}")
    
    print(f"Result: {'✓ PASSED' if has_grad else '✗ FAILED'}")
    return has_grad


def test_adaptive_gating():
    """Test adaptive gating with time embedding."""
    print("\n" + "=" * 60)
    print("Test: Adaptive Gating")
    print("=" * 60)
    
    wvt = WaveletEmbedding(n_scales=4, wavelet='db4')
    
    image = torch.randn(2, 1, 64, 64)
    t_emb = torch.randn(2, 64)
    
    # Without gating
    output_no_gate = wvt(image, t_emb=None)
    
    # With gating
    output_gated = wvt(image, t_emb=t_emb)
    
    # Shapes should be the same
    shape_same = output_no_gate.shape == output_gated.shape
    
    # Values should be different
    values_different = not torch.allclose(output_no_gate, output_gated)
    
    print(f"Without gating: {output_no_gate.shape}")
    print(f"With gating: {output_gated.shape}")
    print(f"Shapes match: {shape_same}")
    print(f"Values different: {values_different}")
    
    passed = shape_same and values_different
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_wavelet_kan_layer():
    """Test Wavelet KAN layer."""
    print("\n" + "=" * 60)
    print("Test: Wavelet KAN Layer")
    print("=" * 60)
    
    layer = WaveletKANLayer(in_features=32, out_features=16, n_wavelets=16)
    
    x = torch.randn(2, 100, 32)
    y = layer(x)
    
    expected_shape = (2, 100, 16)
    shape_correct = y.shape == expected_shape
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Expected: {expected_shape}")
    
    # Test gradient
    loss = y.pow(2).sum()
    loss.backward()
    
    has_grad = layer.wavelet_weights.grad is not None
    
    print(f"Gradient exists: {has_grad}")
    
    passed = shape_correct and has_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_different_wavelets():
    """Test with different wavelet types."""
    print("\n" + "=" * 60)
    print("Test: Different Wavelet Types")
    print("=" * 60)
    
    wavelets = ['haar', 'db4', 'sym4', 'coif1']
    image = torch.randn(1, 1, 64, 64)
    
    all_passed = True
    for wavelet in wavelets:
        try:
            wvt = WaveletEmbedding(n_scales=3, wavelet=wavelet)
            output = wvt.decompose(image)
            print(f"  {wavelet}: {output.shape} ✓")
        except Exception as e:
            print(f"  {wavelet}: FAILED - {e}")
            all_passed = False
    
    print(f"Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WaveletEmbedding Unit Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Decomposition Shape", test_decomposition_shape()))
    results.append(("Multi-scale Content", test_multi_scale_content()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Adaptive Gating", test_adaptive_gating()))
    results.append(("Wavelet KAN Layer", test_wavelet_kan_layer()))
    results.append(("Different Wavelets", test_different_wavelets()))
    
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
