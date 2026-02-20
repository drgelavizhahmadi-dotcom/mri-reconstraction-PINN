"""Hard data consistency layer for MRI reconstruction.

Enforces that measured k-space samples remain unchanged during reconstruction
while allowing the network to fill in missing samples.
"""

import torch
import torch.nn as nn


class ComplexDataConsistency(nn.Module):
    """Hard data consistency constraint for MRI reconstruction.
    
    Enforces the constraint:
        k_dc = mask * k_measured + (1 - mask) * k_pred
    
    Where:
        - mask=1 locations use measured k-space data
        - mask=0 locations use predicted k-space data
    
    Supports both singlecoil and multicoil reconstructions. For multicoil,
    coil sensitivity maps must be provided.
    
    Args:
        spatial_dims: Spatial dimensions for FFT (default: (-2, -1) for 2D)
    
    Example:
        >>> dc = ComplexDataConsistency()
        >>> # Singlecoil
        >>> image_dc = dc(image_pred, k_measured, mask)
        >>> # Multicoil  
        >>> image_dc = dc(image_pred, k_measured, mask, coil_sens)
    """
    
    def __init__(self, spatial_dims: tuple = (-2, -1)):
        super().__init__()
        self.spatial_dims = spatial_dims
    
    @staticmethod
    def _to_complex(tensor: torch.Tensor) -> torch.Tensor:
        """Convert real/imag representation to complex tensor.
        
        Args:
            tensor: Real tensor with shape [..., 2] where last dim is (real, imag)
            
        Returns:
            Complex tensor with shape [...]
        """
        if tensor.shape[-1] != 2:
            raise ValueError(f"Expected last dimension to be 2, got {tensor.shape[-1]}")
        
        # Use view_as_complex - requires contiguous memory layout
        # Input shape: [..., 2] -> Output shape: [...]
        return torch.view_as_complex(tensor.contiguous())
    
    @staticmethod
    def _from_complex(tensor: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor to real/imag representation.
        
        Args:
            tensor: Complex tensor with shape [...]
            
        Returns:
            Real tensor with shape [..., 2] where last dim is (real, imag)
        """
        # Use view_as_real - Output shape: [..., 2]
        return torch.view_as_real(tensor)
    
    def _image_to_kspace(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image to k-space using centered FFT.
        
        Args:
            image: Complex image tensor
            
        Returns:
            Complex k-space tensor
        """
        # Apply FFT shift, FFT, then shift back
        # Using ortho normalization for energy preservation
        kspace = torch.fft.fftn(
            torch.fft.ifftshift(image, dim=self.spatial_dims),
            dim=self.spatial_dims,
            norm='ortho'
        )
        kspace = torch.fft.fftshift(kspace, dim=self.spatial_dims)
        return kspace
    
    def _kspace_to_image(self, kspace: torch.Tensor) -> torch.Tensor:
        """Convert k-space to image using centered IFFT.
        
        Args:
            kspace: Complex k-space tensor
            
        Returns:
            Complex image tensor
        """
        # Apply IFFT shift, IFFT, then shift back
        image = torch.fft.ifftn(
            torch.fft.ifftshift(kspace, dim=self.spatial_dims),
            dim=self.spatial_dims,
            norm='ortho'
        )
        image = torch.fft.fftshift(image, dim=self.spatial_dims)
        return image
    
    def forward(
        self,
        image_pred: torch.Tensor,
        k_measured: torch.Tensor,
        mask: torch.Tensor,
        coil_sens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply hard data consistency.
        
        Args:
            image_pred: Predicted image [B, 2, H, W] (real/imag) or [B, C, 2, H, W] (multicoil)
            k_measured: Measured k-space [B, 2, H, W] or [B, C, 2, H, W]
            mask: Sampling mask [B, 1, H, W] or [B, C, 1, H, W]
            coil_sens: Coil sensitivity maps [B, C, 2, H, W] for multicoil (optional)
            
        Returns:
            Data consistent image with same shape as image_pred
        """
        is_multicoil = coil_sens is not None
        
        if not is_multicoil:
            # Singlecoil reconstruction
            return self._singlecoil_dc(image_pred, k_measured, mask)
        else:
            # Multicoil reconstruction
            return self._multicoil_dc(image_pred, k_measured, mask, coil_sens)
    
    def _singlecoil_dc(
        self,
        image_pred: torch.Tensor,
        k_measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply data consistency for singlecoil data.
        
        Args:
            image_pred: [B, 2, H, W] predicted image (real, imag)
            k_measured: [B, 2, H, W] measured k-space (real, imag)
            mask: [B, 1, H, W] sampling mask
            
        Returns:
            [B, 2, H, W] data consistent image
        """
        # Convert to complex
        image_complex = self._to_complex(image_pred.permute(0, 2, 3, 1))  # [B, H, W]
        k_measured_complex = self._to_complex(k_measured.permute(0, 2, 3, 1))  # [B, H, W]
        
        # Forward: image -> k-space
        k_pred = self._image_to_kspace(image_complex)  # [B, H, W]
        
        # Apply data consistency: k_dc = mask * k_measured + (1-mask) * k_pred
        # Mask is [B, 1, H, W], need to squeeze or expand
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask_complex = mask.squeeze(1)  # [B, H, W]
        else:
            mask_complex = mask
        
        k_dc = mask_complex * k_measured_complex + (1 - mask_complex) * k_pred
        
        # Backward: k-space -> image
        image_dc = self._kspace_to_image(k_dc)  # [B, H, W]
        
        # Convert back to real/imag format
        image_dc_real = self._from_complex(image_dc)  # [B, H, W, 2]
        image_dc = image_dc_real.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        return image_dc
    
    def _multicoil_dc(
        self,
        image_pred: torch.Tensor,
        k_measured: torch.Tensor,
        mask: torch.Tensor,
        coil_sens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply data consistency for multicoil data.
        
        Args:
            image_pred: [B, 2, H, W] predicted image (real, imag)
            k_measured: [B, C, 2, H, W] measured k-space per coil
            mask: [B, C, 1, H, W] or [B, 1, H, W] sampling mask
            coil_sens: [B, C, 2, H, W] coil sensitivity maps
            
        Returns:
            [B, 2, H, W] data consistent combined image
        """
        batch_size, num_coils = k_measured.shape[0], k_measured.shape[1]
        
        # Convert to complex
        image_complex = self._to_complex(image_pred.permute(0, 2, 3, 1))  # [B, H, W]
        k_measured_complex = self._to_complex(
            k_measured.permute(0, 1, 3, 4, 2)
        )  # [B, C, H, W]
        coil_sens_complex = self._to_complex(
            coil_sens.permute(0, 1, 3, 4, 2)
        )  # [B, C, H, W]
        
        # Expand image to coil dimension and apply sensitivities
        image_coil = image_complex.unsqueeze(1) * coil_sens_complex  # [B, C, H, W]
        
        # Forward: coil images -> coil k-spaces
        k_pred = self._image_to_kspace(image_coil)  # [B, C, H, W]
        
        # Handle mask shape
        if mask.dim() == 4:
            # [B, 1, H, W] or [B, C, H, W]
            if mask.shape[1] == 1:
                mask_complex = mask.unsqueeze(1)  # [B, 1, 1, H, W]
            else:
                mask_complex = mask.unsqueeze(2)  # [B, C, 1, H, W]
        elif mask.dim() == 5:
            # [B, C, 1, H, W]
            mask_complex = mask
        else:
            mask_complex = mask.view(batch_size, num_coils, 1, *mask.shape[-2:])
        
        mask_complex = mask_complex.squeeze(2)  # [B, C, H, W]
        
        # Apply data consistency per coil
        k_dc_coil = mask_complex * k_measured_complex + (1 - mask_complex) * k_pred
        
        # Backward: coil k-spaces -> coil images
        image_dc_coil = self._kspace_to_image(k_dc_coil)  # [B, C, H, W]
        
        # Combine coils using sum-of-squares weighted by conjugate sensitivities
        # SENSE-like combination: (S^H * S)^{-1} * S^H * x
        coil_images = image_dc_coil * coil_sens_complex.conj()  # [B, C, H, W]
        image_dc = coil_images.sum(dim=1)  # [B, H, W]
        
        # Normalize by sum of squared sensitivities
        sens_sum = (coil_sens_complex.abs() ** 2).sum(dim=1)  # [B, H, W]
        image_dc = image_dc / (sens_sum + 1e-8)
        
        # Convert back to real/imag format
        image_dc_real = self._from_complex(image_dc)  # [B, H, W, 2]
        image_dc = image_dc_real.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        return image_dc


def test_singlecoil_data_consistency():
    """Test data consistency for singlecoil reconstruction."""
    print("=" * 60)
    print("Testing Singlecoil Data Consistency")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create test parameters
    batch_size = 2
    h, w = 64, 64
    acceleration = 4
    center_fraction = 0.08
    
    # Create data consistency layer
    dc = ComplexDataConsistency()
    
    # 1. Create random ground truth image
    image_gt = torch.randn(batch_size, 2, h, w)
    print(f"\n1. Created ground truth image: {image_gt.shape}")
    
    # 2. Convert to k-space
    image_gt_complex = dc._to_complex(image_gt.permute(0, 2, 3, 1))
    k_full = dc._image_to_kspace(image_gt_complex)
    print(f"2. Converted to k-space: {k_full.shape}")
    
    # 3. Create 4x undersampling mask
    mask = torch.zeros(batch_size, 1, h, w)
    num_center = int(w * center_fraction)
    center_start = (w - num_center) // 2
    center_end = center_start + num_center
    
    # Keep center lines
    mask[:, :, :, center_start:center_end] = 1.0
    
    # Equispaced lines outside center
    num_remaining = w - num_center
    num_samples = num_remaining // acceleration
    
    left_indices = torch.linspace(0, center_start - 1, num_samples // 2, dtype=torch.long)
    right_indices = torch.linspace(center_end, w - 1, num_samples // 2, dtype=torch.long)
    
    mask[:, :, :, left_indices] = 1.0
    mask[:, :, :, right_indices] = 1.0
    
    mask_ratio = mask.sum() / mask.numel()
    print(f"3. Created 4x mask with center fraction {center_fraction}")
    print(f"   Actual sampling ratio: {mask_ratio:.3f} (~1/{1/mask_ratio:.1f})")
    
    # 4. Apply mask to get measured k-space
    k_measured_complex = k_full * mask.squeeze(1)
    k_measured = dc._from_complex(k_measured_complex).permute(0, 3, 1, 2)
    print(f"4. Applied mask to get measured k-space: {k_measured.shape}")
    
    # 5. Add noise to measured k-space (simulate measurement noise)
    noise = torch.randn_like(k_measured) * 0.1
    k_measured_noisy = k_measured + noise
    print(f"5. Added noise to k-space (sigma=0.1)")
    
    # 6. Create a "predicted" image (e.g., from a neural network)
    # Use zero-filled reconstruction as prediction
    k_noisy_complex = dc._to_complex(k_measured_noisy.permute(0, 2, 3, 1))
    image_noisy = dc._kspace_to_image(k_noisy_complex)
    image_pred = dc._from_complex(image_noisy).permute(0, 3, 1, 2)
    print(f"6. Created predicted image (zero-filled): {image_pred.shape}")
    
    # 7. Apply data consistency
    image_dc = dc(image_pred, k_measured_noisy, mask)
    print(f"7. Applied data consistency: {image_dc.shape}")
    
    # 8. Verify measured samples are preserved
    # Convert back to k-space and check
    image_dc_complex = dc._to_complex(image_dc.permute(0, 2, 3, 1))
    k_dc = dc._image_to_kspace(image_dc_complex)
    
    # Extract measured locations
    measured_mask = mask.squeeze(1).bool()
    k_dc_measured = k_dc[measured_mask]
    k_measured_complex_noisy = dc._to_complex(k_measured_noisy.permute(0, 2, 3, 1))
    k_input_measured = k_measured_complex_noisy[measured_mask]
    
    # Check if measured samples are preserved
    error = torch.abs(k_dc_measured - k_input_measured).max().item()
    print(f"\n8. Verification:")
    print(f"   Max error at measured locations: {error:.2e}")
    
    if error < 1e-5:
        print("   ✓ Measured k-space samples are preserved!")
    else:
        print("   ✗ ERROR: Measured samples were modified!")
    
    # Additional checks
    print(f"\nAdditional statistics:")
    print(f"   Input k-space range: [{k_measured_noisy.min():.3f}, {k_measured_noisy.max():.3f}]")
    print(f"   DC output range: [{image_dc.min():.3f}, {image_dc.max():.3f}]")
    
    return error < 1e-5


def test_multicoil_data_consistency():
    """Test data consistency for multicoil reconstruction."""
    print("\n" + "=" * 60)
    print("Testing Multicoil Data Consistency")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create test parameters
    batch_size = 1
    num_coils = 8
    h, w = 64, 64
    acceleration = 4
    
    # Create data consistency layer
    dc = ComplexDataConsistency()
    
    # 1. Create random ground truth image
    image_gt = torch.randn(batch_size, 2, h, w)
    print(f"\n1. Created ground truth image: {image_gt.shape}")
    
    # 2. Create synthetic coil sensitivities (Gaussian-like)
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing='ij'
    )
    
    coil_sens = []
    for c in range(num_coils):
        # Create different sensitivity patterns for each coil
        angle = 2 * torch.pi * c / num_coils
        x_offset = 0.5 * torch.cos(angle)
        y_offset = 0.5 * torch.sin(angle)
        
        sensitivity = torch.exp(-((x_grid - x_offset)**2 + (y_grid - y_offset)**2) / 0.5)
        phase = torch.exp(1j * torch.pi * (x_grid * torch.cos(angle) + y_grid * torch.sin(angle)))
        
        coil_sens_complex = sensitivity * phase
        coil_sens.append(dc._from_complex(coil_sens_complex))
    
    coil_sens = torch.stack(coil_sens, dim=0)  # [C, H, W, 2]
    coil_sens = coil_sens.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [B, C, H, W, 2]
    coil_sens = coil_sens.permute(0, 1, 4, 2, 3)  # [B, C, 2, H, W]
    print(f"2. Created {num_coils} coil sensitivities: {coil_sens.shape}")
    
    # 3. Forward: image -> multicoil k-space
    image_gt_complex = dc._to_complex(image_gt.permute(0, 2, 3, 1))
    coil_sens_complex = dc._to_complex(coil_sens.permute(0, 1, 3, 4, 2))
    
    image_coil = image_gt_complex.unsqueeze(1) * coil_sens_complex
    k_full_coil = dc._image_to_kspace(image_coil)
    
    k_full = dc._from_complex(k_full_coil).permute(0, 1, 4, 2, 3)  # [B, C, 2, H, W]
    print(f"3. Converted to multicoil k-space: {k_full.shape}")
    
    # 4. Create and apply mask
    mask = torch.zeros(batch_size, num_coils, 1, h, w)
    num_center = int(w * 0.08)
    center_start = (w - num_center) // 2
    center_end = center_start + num_center
    mask[:, :, :, :, center_start:center_end] = 1.0
    
    num_remaining = w - num_center
    num_samples = num_remaining // acceleration
    left_indices = torch.linspace(0, center_start - 1, num_samples // 2, dtype=torch.long)
    right_indices = torch.linspace(center_end, w - 1, num_samples // 2, dtype=torch.long)
    mask[:, :, :, :, left_indices] = 1.0
    mask[:, :, :, :, right_indices] = 1.0
    
    print(f"4. Created undersampling mask: {mask.shape}")
    
    # 5. Apply mask and add noise
    k_measured = k_full * mask
    noise = torch.randn_like(k_measured) * 0.05
    k_measured_noisy = k_measured + noise
    print(f"5. Applied mask and added noise")
    
    # 6. Create predicted image (zero-filled RSS)
    k_noisy_complex = dc._to_complex(k_measured_noisy.permute(0, 1, 3, 4, 2))
    image_coil_noisy = dc._kspace_to_image(k_noisy_complex)
    
    # RSS combination
    image_pred_complex = torch.sqrt((image_coil_noisy.abs() ** 2).sum(dim=1))
    image_pred = dc._from_complex(image_pred_complex).permute(0, 3, 1, 2)
    print(f"6. Created predicted image (RSS): {image_pred.shape}")
    
    # 7. Apply data consistency
    image_dc = dc(image_pred, k_measured_noisy, mask, coil_sens)
    print(f"7. Applied data consistency: {image_dc.shape}")
    
    # 8. Verify measured samples are preserved
    # Need to convert back through the multicoil forward model
    image_dc_complex = dc._to_complex(image_dc.permute(0, 2, 3, 1))
    image_dc_coil = image_dc_complex.unsqueeze(1) * coil_sens_complex
    k_dc_coil = dc._image_to_kspace(image_dc_coil)
    
    # Check measured locations
    mask_complex = mask.squeeze(2).bool()  # [B, C, H, W]
    k_dc_measured = k_dc_coil[mask_complex]
    k_input_measured = k_noisy_complex[mask_complex]
    
    error = torch.abs(k_dc_measured - k_input_measured).max().item()
    print(f"\n8. Verification:")
    print(f"   Max error at measured locations: {error:.2e}")
    
    if error < 1e-5:
        print("   ✓ Measured k-space samples are preserved!")
    else:
        print("   ✗ ERROR: Measured samples were modified!")
    
    return error < 1e-5


def test_gradient_flow():
    """Test that gradients flow through data consistency layer."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    batch_size = 1
    h, w = 64, 64
    
    dc = ComplexDataConsistency()
    
    # Create inputs with gradient tracking
    image_pred = torch.randn(batch_size, 2, h, w, requires_grad=True)
    
    # Create measured k-space from ground truth
    image_gt = torch.randn(batch_size, 2, h, w)
    image_gt_complex = dc._to_complex(image_gt.permute(0, 2, 3, 1))
    k_measured_complex = dc._image_to_kspace(image_gt_complex)
    k_measured = dc._from_complex(k_measured_complex).permute(0, 3, 1, 2)
    
    # Create mask
    mask = torch.zeros(batch_size, 1, h, w)
    mask[:, :, :, 16:48] = 1.0
    
    # Forward pass
    image_dc = dc(image_pred, k_measured, mask)
    
    # Compute loss and backward
    loss = image_dc.pow(2).sum()
    loss.backward()
    
    has_grad = image_pred.grad is not None and image_pred.grad.abs().sum() > 0
    
    print(f"\nGradient check:")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradient exists: {has_grad}")
    print(f"   Gradient sum: {image_pred.grad.abs().sum().item():.4f}")
    
    if has_grad:
        print("   ✓ Gradients flow through data consistency!")
    else:
        print("   ✗ ERROR: No gradients!")
    
    return has_grad


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ComplexDataConsistency Unit Tests")
    print("=" * 60)
    
    # Run all tests
    results = []
    
    results.append(("Singlecoil DC", test_singlecoil_data_consistency()))
    results.append(("Multicoil DC", test_multicoil_data_consistency()))
    results.append(("Gradient Flow", test_gradient_flow()))
    
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
