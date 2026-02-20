"""Simplified Bloch model for multi-echo MRI data consistency.

This module implements a physics-based loss that enforces T2* decay
consistency across multiple echo times (TEs).
"""

import torch
import torch.nn as nn


class SimplifiedBlochLoss(nn.Module):
    """T2* decay consistency loss for multi-echo MRI data.
    
    Enforces the physical constraint that signal intensity follows
    exponential T2* decay across echo times:
        S(TE) = S0 * exp(-TE / T2*)
    
    The loss is computed as:
        Loss = ||S_measured(TE) - S0 * exp(-TE / T2*)||^2
    
    T2* is treated as a learnable parameter initialized to a typical
    brain tissue value of 40ms.
    
    Args:
        initial_t2_star: Initial T2* value in milliseconds (default: 40.0)
        te_unit: Unit for TE values ('ms' or 's', default: 'ms')
        reduction: Loss reduction method ('mean', 'sum', 'none', default: 'mean')
        
    Example:
        >>> loss_fn = SimplifiedBlochLoss(initial_t2_star=40.0)
        >>> images = torch.randn(2, 4, 64, 64)  # [B, n_echoes, H, W]
        >>> TEs = torch.tensor([10.0, 30.0, 50.0, 70.0])  # ms
        >>> loss = loss_fn(images, TEs)
        >>> loss.backward()
        >>> print(f"Learned T2*: {loss_fn.get_t2_star():.2f} ms")
    """
    
    def __init__(
        self,
        initial_t2_star: float = 40.0,
        te_unit: str = 'ms',
        reduction: str = 'mean',
    ):
        super().__init__()
        
        # Initialize T2* as a learnable parameter (in milliseconds)
        self.initial_t2_star = initial_t2_star
        self.t2_star = nn.Parameter(torch.tensor(initial_t2_star, dtype=torch.float32))
        
        self.te_unit = te_unit
        self.reduction = reduction
        
        # For numerical stability, ensure T2* stays positive
        self.eps = 1e-6
        
        print(f"SimplifiedBlochLoss initialized:")
        print(f"  Initial T2*: {initial_t2_star} {te_unit}")
        print(f"  TE unit: {te_unit}")
        print(f"  Reduction: {reduction}")
    
    def get_t2_star(self) -> float:
        """Get current T2* value (always positive).
        
        Returns:
            Current T2* value in the original units
        """
        return torch.clamp(self.t2_star, min=self.eps).item()
    
    def estimate_s0(
        self,
        images: torch.Tensor,
        tes: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate S0 (proton density) from first echo.
        
        S0 is estimated by extrapolating from the first echo back to TE=0:
            S0 = S(TE1) * exp(TE1 / T2*)
        
        Args:
            images: Measured images [B, n_echoes, H, W]
            tes: Echo times [n_echoes] in self.te_unit
            
        Returns:
            Estimated S0 map [B, 1, H, W]
        """
        # Get first echo
        first_echo = images[:, 0:1, :, :]  # [B, 1, H, W]
        first_te = tes[0]  # scalar
        
        # Ensure T2* is positive
        t2_star_safe = torch.clamp(self.t2_star, min=self.eps)
        
        # Extrapolate to TE=0: S0 = S(TE1) * exp(TE1 / T2*)
        # Note: exp(TE1/T2*) > 1, so this amplifies the signal
        s0 = first_echo * torch.exp(first_te / t2_star_safe)
        
        return s0
    
    def estimate_s0_rls(
        self,
        images: torch.Tensor,
        tes: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate S0 using recursive least squares across all echoes.
        
        Fits S0 and T2* jointly by solving:
            log(S(TE)) = log(S0) - TE / T2*
        
        This is more robust than single-echo estimation when multiple
        echoes are available.
        
        Args:
            images: Measured images [B, n_echoes, H, W]
            tes: Echo times [n_echoes] in self.te_unit
            
        Returns:
            Estimated S0 map [B, 1, H, W]
        """
        # Avoid log(0)
        images_safe = torch.clamp(images, min=self.eps)
        
        # Take log: log(S) = log(S0) - TE/T2*
        log_images = torch.log(images_safe)  # [B, n_echoes, H, W]
        
        # Linear regression setup
        # X = [1, -TE] (design matrix)
        # y = log(S)
        # Solve for [log(S0), 1/T2*]
        
        n_echoes = len(tes)
        tes_expanded = tes.view(1, n_echoes, 1, 1)  # [1, n_echoes, 1, 1]
        
        # X^T X and X^T y
        # For simple linear regression: y = a + b*x
        # a = log(S0), b = -1/T2*, x = TE
        
        x = tes  # [n_echoes]
        y = log_images  # [B, n_echoes, H, W]
        
        # Compute sums over echoes
        sum_1 = n_echoes
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        sum_y = y.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        sum_xy = (x.view(1, n_echoes, 1, 1) * y).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Solve normal equations
        # [sum_1   sum_x ] [a] = [sum_y ]
        # [sum_x   sum_x2] [b]   [sum_xy]
        
        det = sum_1 * sum_x2 - sum_x ** 2 + self.eps
        
        a = (sum_x2 * sum_y - sum_x * sum_xy) / det  # [B, 1, H, W]
        
        # S0 = exp(a)
        s0 = torch.exp(a)
        
        return s0
    
    def compute_expected_decay(
        self,
        s0: torch.Tensor,
        tes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expected signal decay based on T2* model.
        
        Computes: S_expected(TE) = S0 * exp(-TE / T2*)
        
        Args:
            s0: Proton density map [B, 1, H, W] or [B, n_echoes, H, W]
            tes: Echo times [n_echoes] in self.te_unit
            
        Returns:
            Expected images [B, n_echoes, H, W]
        """
        # Ensure T2* is positive
        t2_star_safe = torch.clamp(self.t2_star, min=self.eps)
        
        # Compute exponential decay: exp(-TE / T2*)
        # tes: [n_echoes] -> [1, n_echoes, 1, 1]
        tes_expanded = tes.view(1, -1, 1, 1)
        
        decay = torch.exp(-tes_expanded / t2_star_safe)  # [1, n_echoes, 1, 1]
        
        # S_expected = S0 * decay
        if s0.dim() == 4 and s0.shape[1] == 1:
            # S0 is [B, 1, H, W], broadcast across echoes
            expected = s0 * decay  # [B, n_echoes, H, W]
        else:
            # S0 already has echo dimension
            expected = s0 * decay
        
        return expected
    
    def forward(
        self,
        images: torch.Tensor,
        tes: torch.Tensor,
        method: str = 'first_echo',
    ) -> torch.Tensor:
        """Compute T2* decay consistency loss.
        
        Args:
            images: Measured images [B, n_echoes, H, W]
            tes: Echo times [n_echoes] in self.te_unit
            method: S0 estimation method ('first_echo' or 'rls')
            
        Returns:
            Scalar loss value (or per-sample loss if reduction='none')
        """
        # Check number of echoes
        n_echoes = images.shape[1]
        
        if n_echoes < 2:
            # Single-echo data: no T2* constraint applicable
            return torch.tensor(0.0, device=images.device, requires_grad=True)
        
        # Ensure tes is on same device as images
        tes = tes.to(images.device)
        
        # Estimate S0
        if method == 'rls':
            s0 = self.estimate_s0_rls(images, tes)
        else:
            s0 = self.estimate_s0(images, tes)
        
        # Compute expected signal
        expected = self.compute_expected_decay(s0, tes)
        
        # Compute loss: ||S_measured - S_expected||^2
        # Use MSE loss
        diff = images - expected
        
        if self.reduction == 'mean':
            loss = (diff ** 2).mean()
        elif self.reduction == 'sum':
            loss = (diff ** 2).sum()
        elif self.reduction == 'none':
            loss = (diff ** 2).mean(dim=[1, 2, 3])  # [B]
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        
        return loss
    
    def predict_t2_star_map(
        self,
        images: torch.Tensor,
        tes: torch.Tensor,
    ) -> torch.Tensor:
        """Predict spatially-varying T2* map from multi-echo data.
        
        Fits T2* at each spatial location independently.
        
        Args:
            images: Measured images [B, n_echoes, H, W]
            tes: Echo times [n_echoes]
            
        Returns:
            T2* map [B, 1, H, W] in the same units as tes
        """
        n_echoes = images.shape[1]
        
        if n_echoes < 2:
            raise ValueError("At least 2 echoes required for T2* mapping")
        
        # Avoid log(0)
        images_safe = torch.clamp(images, min=self.eps)
        
        # Take log: log(S) = log(S0) - TE/T2*
        log_images = torch.log(images_safe)  # [B, n_echoes, H, W]
        
        # Linear regression for -1/T2*
        # y = log(S), x = TE
        # slope = -1/T2*
        
        x = tes.to(images.device)  # [n_echoes]
        y = log_images  # [B, n_echoes, H, W]
        
        # Compute regression coefficients
        n = n_echoes
        sum_x = x.sum()
        sum_x2 = (x ** 2).sum()
        
        det = n * sum_x2 - sum_x ** 2 + self.eps
        
        # Slope = (n*sum(xy) - sum(x)*sum(y)) / det
        sum_xy = (x.view(1, n_echoes, 1, 1) * y).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        sum_y = y.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        slope = (n * sum_xy - sum_x * sum_y) / det  # [B, 1, H, W]
        
        # T2* = -1 / slope
        t2_star_map = -1.0 / (slope + self.eps)
        
        # Clamp to reasonable range (e.g., 1-200 ms for brain)
        t2_star_map = torch.clamp(t2_star_map, min=1.0, max=200.0)
        
        return t2_star_map


def test_single_echo():
    """Test that single-echo data returns 0 loss."""
    print("\n" + "=" * 60)
    print("Test: Single-echo data returns zero loss")
    print("=" * 60)
    
    loss_fn = SimplifiedBlochLoss()
    
    # Single echo data
    images = torch.randn(2, 1, 64, 64)
    tes = torch.tensor([10.0])
    
    loss = loss_fn(images, tes)
    
    print(f"Images shape: {images.shape}")
    print(f"TEs: {tes}")
    print(f"Loss: {loss.item()}")
    
    passed = abs(loss.item()) < 1e-6
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_t2_convergence():
    """Test that loss decreases with optimization."""
    print("\n" + "=" * 60)
    print("Test: T2* parameter converges with optimization")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create synthetic multi-echo data with known T2*
    true_t2_star = 50.0  # ms
    s0_true = torch.ones(2, 1, 64, 64) * 100.0
    tes = torch.tensor([10.0, 30.0, 50.0, 70.0])
    
    # Generate true signal
    decay = torch.exp(-tes.view(1, -1, 1, 1) / true_t2_star)
    images_true = s0_true * decay
    
    # Add noise
    images_noisy = images_true + torch.randn_like(images_true) * 2.0
    
    print(f"True T2*: {true_t2_star} ms")
    print(f"Initial estimated T2*: {40.0} ms")
    
    # Create loss function
    loss_fn = SimplifiedBlochLoss(initial_t2_star=40.0)
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=1.0)
    
    # Optimize
    losses = []
    t2_stars = []
    
    for step in range(100):
        optimizer.zero_grad()
        loss = loss_fn(images_noisy, tes)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        t2_stars.append(loss_fn.get_t2_star())
    
    final_t2_star = loss_fn.get_t2_star()
    print(f"Final learned T2*: {final_t2_star:.2f} ms")
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Check convergence
    t2_error = abs(final_t2_star - true_t2_star)
    loss_decreased = losses[-1] < losses[0]
    
    print(f"T2* error: {t2_error:.2f} ms")
    print(f"Loss decreased: {loss_decreased} ({losses[0]:.2f} -> {losses[-1]:.2f})")
    
    passed = t2_error < 10.0 and loss_decreased
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_s0_estimation():
    """Test S0 estimation from first echo."""
    print("\n" + "=" * 60)
    print("Test: S0 estimation accuracy")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    true_t2_star = 40.0
    true_s0 = torch.ones(2, 1, 64, 64) * 100.0
    tes = torch.tensor([20.0, 40.0, 60.0, 80.0])
    
    # Generate true signal
    decay = torch.exp(-tes.view(1, -1, 1, 1) / true_t2_star)
    images = true_s0 * decay
    
    loss_fn = SimplifiedBlochLoss(initial_t2_star=true_t2_star)
    
    # Estimate S0
    s0_estimated = loss_fn.estimate_s0(images, tes)
    
    mean_error = (s0_estimated - true_s0).abs().mean().item()
    
    print(f"True S0: {true_s0.mean().item():.2f}")
    print(f"Estimated S0: {s0_estimated.mean().item():.2f}")
    print(f"Mean absolute error: {mean_error:.4f}")
    
    passed = mean_error < 1.0
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_gradient_flow():
    """Test that gradients flow properly."""
    print("\n" + "=" * 60)
    print("Test: Gradient flow")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create model with learnable T2*
    loss_fn = SimplifiedBlochLoss()
    
    # Create images that require gradients
    images = torch.randn(2, 4, 64, 64, requires_grad=True)
    tes = torch.tensor([10.0, 30.0, 50.0, 70.0])
    
    # Forward
    loss = loss_fn(images, tes)
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_image_grad = images.grad is not None and images.grad.abs().sum() > 0
    has_t2_grad = loss_fn.t2_star.grad is not None and loss_fn.t2_star.grad.abs() > 0
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Image gradients exist: {has_image_grad}")
    print(f"T2* gradient exists: {has_t2_grad}")
    print(f"T2* gradient value: {loss_fn.t2_star.grad.item():.6f}")
    
    passed = has_image_grad and has_t2_grad
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


def test_t2_star_mapping():
    """Test spatial T2* mapping."""
    print("\n" + "=" * 60)
    print("Test: Spatial T2* mapping")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create data with spatially varying T2*
    h, w = 32, 32
    
    # Create T2* map (gradient from 30 to 70 ms)
    t2_star_map = torch.linspace(30, 70, h).view(1, 1, h, 1).expand(1, 1, h, w)
    
    s0 = torch.ones(1, 1, h, w) * 100.0
    tes = torch.tensor([10.0, 30.0, 50.0, 70.0])
    
    # Generate signal
    images = []
    for te in tes:
        decay = torch.exp(-te / t2_star_map)
        image = s0 * decay
        images.append(image)
    images = torch.cat(images, dim=1)  # [1, 4, H, W]
    
    # Predict T2* map
    loss_fn = SimplifiedBlochLoss()
    t2_star_predicted = loss_fn.predict_t2_star_map(images, tes)
    
    mean_error = (t2_star_predicted - t2_star_map).abs().mean().item()
    
    print(f"True T2* range: [{t2_star_map.min():.1f}, {t2_star_map.max():.1f}] ms")
    print(f"Predicted T2* range: [{t2_star_predicted.min():.1f}, {t2_star_predicted.max():.1f}] ms")
    print(f"Mean absolute error: {mean_error:.2f} ms")
    
    passed = mean_error < 5.0
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SimplifiedBlochLoss Unit Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Single-echo", test_single_echo()))
    results.append(("T2* convergence", test_t2_convergence()))
    results.append(("S0 estimation", test_s0_estimation()))
    results.append(("Gradient flow", test_gradient_flow()))
    results.append(("T2* mapping", test_t2_star_mapping()))
    
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
