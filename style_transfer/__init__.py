from .neural_style import NeuralStyleTransfer
from .fast_style import FastStyleNetwork
from .gan import StyleGAN
from .presets import STYLE_PRESETS, get_preset

__all__ = [
    "NeuralStyleTransfer",
    "FastStyleNetwork",
    "StyleGAN",
    "STYLE_PRESETS",
    "get_preset",
]
