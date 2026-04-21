import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List


# Named VGG19 layers used throughout the literature
CONTENT_LAYERS_DEFAULT = ["relu4_2"]
STYLE_LAYERS_DEFAULT   = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

_VGG19_LAYER_NAMES = [
    "conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
    "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
    "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3",
    "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4",
    "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5",
]


class VGGFeatureExtractor(nn.Module):
    """
    Frozen VGG19 that returns intermediate feature maps by name.
    Replaces MaxPool with AvgPool for smoother gradients (Gatys et al. recommendation).
    """

    def __init__(
        self,
        layers: List[str],
        use_avg_pool: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.requested = set(layers)

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices: nn.ModuleDict = nn.ModuleDict()
        self._layer_to_slice: Dict[str, str] = {}

        slice_start = 0
        slice_idx = 0
        for child_idx, (layer_name, module) in enumerate(
            zip(_VGG19_LAYER_NAMES, vgg.children())
        ):
            if use_avg_pool and isinstance(module, nn.MaxPool2d):
                module = nn.AvgPool2d(kernel_size=2, stride=2)

            if layer_name in self.requested:
                # build a slice up to and including this layer
                slice_layers = list(vgg.children())[slice_start : child_idx + 1]
                if use_avg_pool:
                    slice_layers = [
                        nn.AvgPool2d(kernel_size=2, stride=2) if isinstance(m, nn.MaxPool2d) else m
                        for m in slice_layers
                    ]
                self.slices[f"slice{slice_idx}"] = nn.Sequential(*slice_layers)
                self._layer_to_slice[layer_name] = f"slice{slice_idx}"
                slice_start = child_idx + 1
                slice_idx += 1

        for param in self.parameters():
            param.requires_grad_(False)

        self.to(device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        h = x
        for slice_name, slice_module in self.slices.items():
            h = slice_module(h)
            layer_name = next(
                k for k, v in self._layer_to_slice.items() if v == slice_name
            )
            features[layer_name] = h
        return features
