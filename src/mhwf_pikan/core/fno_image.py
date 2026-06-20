"""
Image-domain FNO reconstructor.

Takes zero-filled complex image [B,2,H,W] as input; works entirely in
image space (no k-space path inside the model). Output is either:
  - magnitude [B,1,H,W]  (output_complex=False, changes A & B)
  - complex   [B,2,H,W]  (output_complex=True,  change C — fed into DC layer)

Architecture: lift → N×FNOBlock → project
FNOBlock: spectral conv (low-mode Fourier mult) + 1×1 skip + GN + GELU + residual

KAN layers are intentionally omitted here: the main bottleneck is the broken
normalization, not the activation function. KAN can be added back after the
pipeline is verified correct.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── spectral layer ──────────────────────────────────

class SpectralConv2d(nn.Module):
    """Fourier-space multiplicative layer (real-stored complex weights)."""

    def __init__(self, in_ch: int, out_ch: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        self.in_ch = in_ch
        self.out_ch = out_ch
        scale = 1.0 / (in_ch * out_ch)
        # weights shape: [in_ch, out_ch, modes, modes, 2] — last dim is (re, im)
        self.w = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.in_ch

        x_ft = torch.fft.rfft2(x, norm="ortho")                  # [B,C,H,W//2+1] complex

        mh = min(self.modes, H)
        mw = min(self.modes, W // 2 + 1)

        wr = self.w[:, :, :mh, :mw, 0]  # [in_ch, out_ch, mh, mw]
        wi = self.w[:, :, :mh, :mw, 1]

        xr = x_ft.real[:, :, :mh, :mw]  # [B, in_ch, mh, mw]
        xi = x_ft.imag[:, :, :mh, :mw]

        # Complex matmul: (xr + i*xi)(wr + i*wi) = (xr*wr - xi*wi) + i*(xr*wi + xi*wr)
        out_r = (torch.einsum("bimn,iomn->bomn", xr, wr)
                 - torch.einsum("bimn,iomn->bomn", xi, wi))
        out_i = (torch.einsum("bimn,iomn->bomn", xr, wi)
                 + torch.einsum("bimn,iomn->bomn", xi, wr))

        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.complex64, device=x.device)
        out_ft[:, :, :mh, :mw] = torch.complex(out_r, out_i)

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")  # [B,out_ch,H,W]


# ─────────────────────────── FNO block ───────────────────────────────────────

class FNOBlock(nn.Module):
    """Single FNO layer: spectral path + 1×1 skip + GroupNorm + GELU + residual."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.skip = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(min(8, width), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.norm(self.spectral(x) + self.skip(x)))


# ─────────────────────────── main model ──────────────────────────────────────

class FNOImageReconstructor(nn.Module):
    """
    Image-domain FNO reconstructor for accelerated MRI.

    Args:
        modes          : number of Fourier modes per dimension (default 16)
        width          : channel width (default 64)
        n_fno          : number of FNO blocks (default 4)
        output_complex : False → 1-ch magnitude (changes A,B)
                         True  → 2-ch complex for DC layer (change C)

    Input : zero-filled complex image [B, 2, H, W]  (real/imag, scale-normalised)
    Output: [B, 1, H, W] or [B, 2, H, W]
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_fno: int = 4,
        output_complex: bool = False,
    ) -> None:
        super().__init__()
        self.output_complex = output_complex
        out_ch = 2 if output_complex else 1

        self.lift = nn.Conv2d(2, width, 1)
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes) for _ in range(n_fno)]
        )
        self.project = nn.Sequential(
            nn.Conv2d(width, width // 2, 1),
            nn.GELU(),
            nn.Conv2d(width // 2, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 2, H, W] zero-filled complex (real,imag channels)."""
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.project(x)

    @staticmethod
    def count_params(model: "FNOImageReconstructor") -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
