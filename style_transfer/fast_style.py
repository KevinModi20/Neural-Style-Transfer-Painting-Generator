"""
Fast Feed-Forward Style Transfer (Johnson et al., 2016).

A ResNet-style transformation network is trained once per artistic style
using a perceptual loss.  At inference the network runs in a single forward
pass — orders of magnitude faster than optimization-based NST.

Architecture
------------
  Encoder  : 3 strided conv blocks (downsampling)
  Residual : N residual blocks (feature refinement)
  Decoder  : 3 fractional-stride conv blocks (upsampling)

All conv layers use instance normalization (Ulyanov et al., 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvNorm(nn.Sequential):
    """Conv2d + InstanceNorm2d + optional ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        upsample: bool = False,
        relu: bool = True,
    ) -> None:
        layers: list[nn.Module] = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, padding_mode="reflect")
        )
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    """Two ConvNorm blocks with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNorm(channels, channels, 3, padding=1),
            ConvNorm(channels, channels, 3, padding=1, relu=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Transformation network
# ---------------------------------------------------------------------------

class TransformNet(nn.Module):
    """
    Image-to-image network for real-time style transfer.

    Parameters
    ----------
    n_res_blocks : int
        Number of residual blocks (default 5, paper uses 5 for 256-px, 9 for 512+).
    base_channels : int
        Width of the initial encoding layer.
    """

    def __init__(self, n_res_blocks: int = 5, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            ConvNorm(3,     c,     9, stride=1, padding=4),
            ConvNorm(c,     c * 2, 3, stride=2, padding=1),
            ConvNorm(c * 2, c * 4, 3, stride=2, padding=1),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(c * 4) for _ in range(n_res_blocks)])
        self.decoder = nn.Sequential(
            ConvNorm(c * 4, c * 2, 3, stride=1, padding=1, upsample=True),
            ConvNorm(c * 2, c,     3, stride=1, padding=1, upsample=True),
            nn.Conv2d(c, 3, 9, stride=1, padding=4, padding_mode="reflect"),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tanh output is in [-1, 1]; scale to approximate ImageNet-normalized range
        return self.decoder(self.residuals(self.encoder(x)))


# ---------------------------------------------------------------------------
# Wrapper with save / load helpers
# ---------------------------------------------------------------------------

class FastStyleNetwork:
    """
    Thin wrapper around TransformNet for inference and checkpoint management.

    Usage
    -----
    net = FastStyleNetwork.from_checkpoint("checkpoints/starry_night.pth")
    output = net.stylize(content_tensor)
    """

    def __init__(
        self,
        n_res_blocks: int = 5,
        base_channels: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.net = TransformNet(n_res_blocks, base_channels).to(device)

    # ------------------------------------------------------------------
    def stylize(self, content: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass. Returns normalized output tensor."""
        self.net.eval()
        with torch.no_grad():
            return self.net(content.to(self.device))

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.net.state_dict()},
            path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device = torch.device("cpu"),
    ) -> "FastStyleNetwork":
        ckpt = torch.load(path, map_location=device)
        fsn = cls(device=device)
        fsn.net.load_state_dict(ckpt["state_dict"])
        return fsn

    # ------------------------------------------------------------------
    # Expose the underlying nn.Module for training loops
    # ------------------------------------------------------------------
    @property
    def module(self) -> TransformNet:
        return self.net
