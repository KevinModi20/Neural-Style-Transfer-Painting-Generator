import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .utils import gram_matrix, total_variation
from .vgg import VGGFeatureExtractor, CONTENT_LAYERS_DEFAULT, STYLE_LAYERS_DEFAULT


class ContentLoss(nn.Module):
    """MSE between content features of generated and target images."""

    def forward(
        self,
        gen_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor],
        layers: List[str],
        weights: Dict[str, float],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(gen_features.values())).device)
        for layer in layers:
            w = weights.get(layer, 1.0)
            loss = loss + w * F.mse_loss(gen_features[layer], target_features[layer].detach())
        return loss


class StyleLoss(nn.Module):
    """MSE between Gram matrices of generated and style images across multiple layers."""

    def forward(
        self,
        gen_features: Dict[str, torch.Tensor],
        style_grams: Dict[str, torch.Tensor],
        layers: List[str],
        weights: Dict[str, float],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(gen_features.values())).device)
        for layer in layers:
            w = weights.get(layer, 1.0)
            g_gen   = gram_matrix(gen_features[layer])
            g_style = style_grams[layer].detach()
            loss = loss + w * F.mse_loss(g_gen, g_style)
        return loss


class TotalVariationLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return total_variation(x)


class PerceptualLoss(nn.Module):
    """
    Combined perceptual loss used for training fast style-transfer networks.
    = content_weight * L_content + style_weight * L_style + tv_weight * L_tv
    """

    def __init__(
        self,
        style_image: torch.Tensor,
        content_layers: List[str] = CONTENT_LAYERS_DEFAULT,
        style_layers: List[str] = STYLE_LAYERS_DEFAULT,
        content_weights: Dict[str, float] = None,
        style_weights: Dict[str, float] = None,
        content_weight: float = 1e0,
        style_weight: float = 1e5,
        tv_weight: float = 1e-6,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        all_layers = list(set(content_layers) | set(style_layers))
        self.vgg = VGGFeatureExtractor(all_layers, device=device)

        self.content_layers  = content_layers
        self.style_layers    = style_layers
        self.content_weights = content_weights or {l: 1.0 for l in content_layers}
        self.style_weights   = style_weights   or {l: 1.0 for l in style_layers}
        self.content_weight  = content_weight
        self.style_weight    = style_weight
        self.tv_weight       = tv_weight

        self.content_loss = ContentLoss()
        self.style_loss   = StyleLoss()
        self.tv_loss      = TotalVariationLoss()

        # Pre-compute style Gram matrices (detached plain tensors, not buffers)
        with torch.no_grad():
            style_feats = self.vgg(style_image.to(device))
        self._style_grams: Dict[str, torch.Tensor] = {
            l: gram_matrix(style_feats[l]).detach() for l in style_layers
        }

    def forward(
        self,
        generated: torch.Tensor,
        content: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gen_feats     = self.vgg(generated)
        content_feats = self.vgg(content)

        l_content = self.content_loss(gen_feats, content_feats, self.content_layers, self.content_weights)
        l_style   = self.style_loss(gen_feats, self._style_grams, self.style_layers, self.style_weights)
        l_tv      = self.tv_loss(generated)

        total = (
            self.content_weight * l_content
            + self.style_weight  * l_style
            + self.tv_weight     * l_tv
        )

        return {
            "total":   total,
            "content": l_content,
            "style":   l_style,
            "tv":      l_tv,
        }
