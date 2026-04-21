"""
Optimization-based Neural Style Transfer (Gatys et al., 2015).

Directly optimizes a generated image by minimizing:
  L = alpha * L_content + beta * L_style + gamma * L_tv

Uses L-BFGS (or optionally Adam) — same image as both content init and
optimizer parameter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from .losses import ContentLoss, StyleLoss, TotalVariationLoss
from .utils import gram_matrix, save_image
from .vgg import (
    CONTENT_LAYERS_DEFAULT,
    STYLE_LAYERS_DEFAULT,
    VGGFeatureExtractor,
)


@dataclass
class NSTConfig:
    content_layers:  List[str]        = field(default_factory=lambda: CONTENT_LAYERS_DEFAULT)
    style_layers:    List[str]        = field(default_factory=lambda: STYLE_LAYERS_DEFAULT)
    content_weights: Dict[str, float] = field(default_factory=dict)
    style_weights:   Dict[str, float] = field(default_factory=dict)
    content_weight:  float = 1e0
    style_weight:    float = 1e6
    tv_weight:       float = 1e-4
    n_steps:         int   = 500
    optimizer:       str   = "lbfgs"   # "lbfgs" | "adam"
    lr:              float = 1.0       # used only for adam
    init:            str   = "content" # "content" | "style" | "random"
    save_every:      int   = 0         # 0 = disabled
    output_dir:      str   = "outputs"


class NeuralStyleTransfer:
    """
    Implements the original Gatys-style optimization loop.

    Usage
    -----
    nst = NeuralStyleTransfer(device)
    result = nst.run(content_tensor, style_tensor, config)
    """

    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        config: NSTConfig = None,
        callback: Optional[Callable[[int, torch.Tensor, Dict], None]] = None,
    ) -> torch.Tensor:
        if config is None:
            config = NSTConfig()

        content = content.to(self.device)
        style   = style.to(self.device)

        all_layers = list(set(config.content_layers) | set(config.style_layers))
        vgg = VGGFeatureExtractor(all_layers, device=self.device)

        content_loss_fn = ContentLoss()
        style_loss_fn   = StyleLoss()
        tv_loss_fn      = TotalVariationLoss()

        # Pre-compute target features
        with torch.no_grad():
            content_feats = vgg(content)
            style_feats   = vgg(style)

        style_grams = {l: gram_matrix(style_feats[l]) for l in config.style_layers}
        c_weights = config.content_weights or {l: 1.0 for l in config.content_layers}
        s_weights = config.style_weights   or {l: 1.0 for l in config.style_layers}

        # Initialize the generated image
        generated = self._init_image(content, style, config.init).requires_grad_(True)

        optimizer = self._build_optimizer(generated, config)
        history: Dict[str, list] = {"total": [], "content": [], "style": [], "tv": []}

        pbar = tqdm(range(1, config.n_steps + 1), desc="NST", dynamic_ncols=True)

        def closure():
            optimizer.zero_grad()
            gen_feats = vgg(generated)

            l_c  = content_loss_fn(gen_feats, content_feats, config.content_layers, c_weights)
            l_s  = style_loss_fn(gen_feats, style_grams, config.style_layers, s_weights)
            l_tv = tv_loss_fn(generated)

            loss = config.content_weight * l_c + config.style_weight * l_s + config.tv_weight * l_tv
            loss.backward()

            history["total"].append(loss.item())
            history["content"].append(l_c.item())
            history["style"].append(l_s.item())
            history["tv"].append(l_tv.item())
            return loss

        for step in pbar:
            if config.optimizer == "lbfgs":
                optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()

            with torch.no_grad():
                generated.clamp_(-3.0, 3.0)

            if history["total"]:
                pbar.set_postfix(
                    loss=f"{history['total'][-1]:.2f}",
                    style=f"{history['style'][-1]:.2f}",
                )

            if config.save_every > 0 and step % config.save_every == 0:
                save_image(generated, f"{config.output_dir}/step_{step:04d}.png")

            if callback:
                callback(step, generated.detach().clone(), history)

        return generated.detach()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_image(
        self, content: torch.Tensor, style: torch.Tensor, mode: str
    ) -> torch.Tensor:
        if mode == "content":
            return content.clone()
        if mode == "style":
            return style.clone()
        # random noise
        return torch.randn_like(content) * 0.01

    def _build_optimizer(self, img: torch.Tensor, config: NSTConfig):
        if config.optimizer == "lbfgs":
            return optim.LBFGS([img], lr=config.lr, max_iter=20)
        return optim.Adam([img], lr=config.lr)
