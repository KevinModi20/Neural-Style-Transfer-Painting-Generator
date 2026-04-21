"""
GAN-based Style Enhancement.

Architecture
------------
Generator  : U-Net with skip connections (encoder–decoder with bottleneck)
Discriminator : PatchGAN (70×70 receptive field)

Loss
----
  G_loss = lambda_adv   * L_adversarial (LSGAN)
         + lambda_perc  * L_perceptual (VGG content + style)
         + lambda_tv    * L_total_variation
  D_loss = LSGAN least-squares objective

The U-Net generator preserves content structure while the adversarial
training drives high-frequency detail toward the target style distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .losses import PerceptualLoss
from .utils import save_image


# ---------------------------------------------------------------------------
# U-Net Generator
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """Conv → BN → LeakyReLU (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int, bn: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not bn)
        ]
        if bn:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose → BN → ReLU + optional skip concat (decoder step)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.block(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x


class UNetGenerator(nn.Module):
    """
    U-Net generator for 256×256 images (8 down/up stages).
    Input/output: 3-channel ImageNet-normalized tensors.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 64) -> None:
        super().__init__()
        # Encoder
        self.e1 = DownBlock(in_ch,       base,      bn=False)  # 128
        self.e2 = DownBlock(base,        base * 2)              # 64
        self.e3 = DownBlock(base * 2,    base * 4)              # 32
        self.e4 = DownBlock(base * 4,    base * 8)              # 16
        self.e5 = DownBlock(base * 8,    base * 8)              # 8
        self.e6 = DownBlock(base * 8,    base * 8)              # 4
        self.e7 = DownBlock(base * 8,    base * 8)              # 2
        self.e8 = DownBlock(base * 8,    base * 8, bn=False)    # 1 (bottleneck)

        # Decoder (in_ch = prev_out + skip_ch)
        self.d1 = UpBlock(base * 8,          base * 8, dropout=0.5)  # 2,  cat → 1024
        self.d2 = UpBlock(base * 8 * 2,      base * 8, dropout=0.5)  # 4,  cat → 1024
        self.d3 = UpBlock(base * 8 * 2,      base * 8, dropout=0.5)  # 8,  cat → 1024
        self.d4 = UpBlock(base * 8 * 2,      base * 8)               # 16, cat → 1024
        self.d5 = UpBlock(base * 8 * 2,      base * 4)               # 32, cat → 512
        self.d6 = UpBlock(base * 4 * 2,      base * 2)               # 64, cat → 256
        self.d7 = UpBlock(base * 2 * 2,      base)                   # 128,cat → 128
        self.out = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8, e7)
        d2 = self.d2(d1,  e6)
        d3 = self.d3(d2,  e5)
        d4 = self.d4(d3,  e4)
        d5 = self.d5(d4,  e3)
        d6 = self.d6(d5,  e2)
        d7 = self.d7(d6,  e1)
        return self.out(d7)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.
    Outputs a map of real/fake scores; each score covers a 70×70 patch.
    """

    def __init__(self, in_ch: int = 6, base: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # in_ch = concat(content, generated_or_style)
            nn.Conv2d(in_ch, base,      4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base,     base * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, content: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([content, image], dim=1)
        return self.model(x)


# ---------------------------------------------------------------------------
# LSGAN losses
# ---------------------------------------------------------------------------

def _lsgan_d_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        F.mse_loss(real_pred, torch.ones_like(real_pred))
        + F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
    )


def _lsgan_g_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(fake_pred, torch.ones_like(fake_pred))


# ---------------------------------------------------------------------------
# StyleGAN trainer
# ---------------------------------------------------------------------------

@dataclass
class GANConfig:
    lambda_adv:  float = 1.0
    lambda_perc: float = 10.0
    lambda_tv:   float = 1e-5
    lr_g:        float = 2e-4
    lr_d:        float = 2e-4
    betas:       Tuple[float, float] = (0.5, 0.999)
    n_epochs:    int   = 50
    save_every:  int   = 5
    output_dir:  str   = "outputs/gan"
    ckpt_dir:    str   = "checkpoints/gan"


class StyleGAN:
    """
    Trains / runs the U-Net + PatchGAN style enhancer.

    Usage (training)
    ----------------
    gan = StyleGAN(style_image, device=device)
    gan.train(dataloader, config)

    Usage (inference)
    -----------------
    gan = StyleGAN.from_checkpoint(ckpt_path, device=device)
    output = gan.generate(content_tensor)
    """

    def __init__(
        self,
        style_image: Optional[torch.Tensor] = None,
        perceptual_config: Optional[dict] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.G = UNetGenerator().to(device)
        self.D = PatchGANDiscriminator().to(device)

        if style_image is not None:
            pc = perceptual_config or {}
            self.perc_loss = PerceptualLoss(
                style_image=style_image,
                device=device,
                **pc,
            )
        else:
            self.perc_loss = None

        self._init_weights(self.G)
        self._init_weights(self.D)

    @staticmethod
    def _init_weights(net: nn.Module) -> None:
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        config: GANConfig = None,
    ) -> Dict[str, List[float]]:
        if config is None:
            config = GANConfig()
        assert self.perc_loss is not None, "Provide style_image to StyleGAN for training."

        opt_G = optim.Adam(self.G.parameters(), lr=config.lr_g, betas=config.betas)
        opt_D = optim.Adam(self.D.parameters(), lr=config.lr_d, betas=config.betas)

        history: Dict[str, List[float]] = {"G": [], "D": [], "perc": []}
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(1, config.n_epochs + 1):
            self.G.train()
            self.D.train()
            g_epoch, d_epoch, p_epoch = 0.0, 0.0, 0.0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.n_epochs}", leave=False)
            for batch in pbar:
                content = batch.to(self.device)
                fake    = self.G(content)

                # ---- Discriminator step ----
                opt_D.zero_grad()
                real_pred = self.D(content, content)        # "real" = content itself
                fake_pred = self.D(content, fake.detach())
                d_loss = _lsgan_d_loss(real_pred, fake_pred)
                d_loss.backward()
                opt_D.step()

                # ---- Generator step ----
                opt_G.zero_grad()
                fake_pred_g = self.D(content, fake)
                adv_loss  = _lsgan_g_loss(fake_pred_g)
                perc_dict = self.perc_loss(fake, content)
                g_loss = (
                    config.lambda_adv  * adv_loss
                    + config.lambda_perc * perc_dict["total"]
                )
                g_loss.backward()
                opt_G.step()

                g_epoch += g_loss.item()
                d_epoch += d_loss.item()
                p_epoch += perc_dict["total"].item()
                pbar.set_postfix(G=f"{g_loss.item():.3f}", D=f"{d_loss.item():.3f}")

            n = len(dataloader)
            history["G"].append(g_epoch / n)
            history["D"].append(d_epoch / n)
            history["perc"].append(p_epoch / n)

            if epoch % config.save_every == 0:
                self.save(Path(config.ckpt_dir) / f"epoch_{epoch:03d}.pth")

        return history

    # ------------------------------------------------------------------
    def generate(self, content: torch.Tensor) -> torch.Tensor:
        self.G.eval()
        with torch.no_grad():
            return self.G(content.to(self.device))

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
            },
            path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device = torch.device("cpu"),
    ) -> "StyleGAN":
        ckpt = torch.load(path, map_location=device)
        gan = cls(device=device)
        gan.G.load_state_dict(ckpt["G"])
        gan.D.load_state_dict(ckpt["D"])
        return gan
