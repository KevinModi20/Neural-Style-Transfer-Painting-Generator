"""
Artistic style presets with tuned hyperparameters.

Each preset captures the visual characteristics of a movement by adjusting
the balance between content fidelity, style strength, and which VGG layers
carry the most weight.

Impressionism  – loose brush strokes, rich colour blends, soft edges.
                 High style weight, strong low-frequency layers.
Cubism         – fragmented geometric planes, multiple viewpoints.
                 Very high style weight, balanced across all layers.
Pointillism    – discrete dots of pure colour, no smooth blending.
                 Highest style weight, focus on early (fine) layers.
Expressionism  – distorted shapes, bold outlines, emotional colour.
                 High content + high style weight.
Watercolour    – translucent washes, soft blooms, paper texture.
                 Moderate style, very low TV (allow some noise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StylePreset:
    name:            str
    description:     str
    # VGG layer selection
    content_layers:  List[str]        = field(default_factory=lambda: ["relu4_2"])
    style_layers:    List[str]        = field(default_factory=lambda: ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"])
    # Per-layer weights
    content_weights: Dict[str, float] = field(default_factory=dict)
    style_weights:   Dict[str, float] = field(default_factory=dict)
    # Global loss weights
    content_weight:  float = 1.0
    style_weight:    float = 1e6
    tv_weight:       float = 1e-4
    # Optimizer settings
    n_steps:         int   = 500
    optimizer:       str   = "lbfgs"
    lr:              float = 1.0
    # Image init mode
    init:            str   = "content"


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

IMPRESSIONISM = StylePreset(
    name="impressionism",
    description=(
        "Soft, painterly brushstrokes with luminous colour mixing. "
        "Inspired by Monet and Renoir."
    ),
    content_layers=["relu4_2"],
    style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
    style_weights={
        "relu1_1": 0.2,
        "relu2_1": 0.5,
        "relu3_1": 1.0,
        "relu4_1": 1.5,
        "relu5_1": 1.0,
    },
    content_weight=1.0,
    style_weight=1e6,
    tv_weight=5e-5,
    n_steps=600,
    optimizer="lbfgs",
    init="content",
)

CUBISM = StylePreset(
    name="cubism",
    description=(
        "Fragmented geometric planes and angular decomposition. "
        "Inspired by Picasso and Braque."
    ),
    content_layers=["relu3_2"],
    style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
    style_weights={
        "relu1_1": 1.0,
        "relu2_1": 1.0,
        "relu3_1": 1.0,
        "relu4_1": 1.0,
        "relu5_1": 1.0,
    },
    content_weight=0.5,
    style_weight=5e6,
    tv_weight=1e-4,
    n_steps=700,
    optimizer="lbfgs",
    init="content",
)

POINTILLISM = StylePreset(
    name="pointillism",
    description=(
        "Pure-colour dot patterns with vibrant optical mixing. "
        "Inspired by Seurat and Signac."
    ),
    content_layers=["relu4_2"],
    style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1"],
    style_weights={
        "relu1_1": 2.0,
        "relu2_1": 2.0,
        "relu3_1": 1.0,
        "relu4_1": 0.5,
    },
    content_weight=1.0,
    style_weight=8e6,
    tv_weight=1e-5,
    n_steps=800,
    optimizer="lbfgs",
    init="content",
)

EXPRESSIONISM = StylePreset(
    name="expressionism",
    description=(
        "Bold distorted forms and emotionally charged colour palettes. "
        "Inspired by Van Gogh and Munch."
    ),
    content_layers=["relu3_2", "relu4_2"],
    style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
    style_weights={
        "relu1_1": 0.5,
        "relu2_1": 1.0,
        "relu3_1": 1.5,
        "relu4_1": 2.0,
        "relu5_1": 1.5,
    },
    content_weight=1.5,
    style_weight=1e6,
    tv_weight=1e-4,
    n_steps=600,
    optimizer="lbfgs",
    init="content",
)

WATERCOLOUR = StylePreset(
    name="watercolour",
    description=(
        "Translucent washes and soft colour blooms on textured paper. "
        "Inspired by Turner and Sargent."
    ),
    content_layers=["relu4_2"],
    style_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"],
    style_weights={
        "relu1_1": 0.1,
        "relu2_1": 0.3,
        "relu3_1": 0.8,
        "relu4_1": 1.2,
        "relu5_1": 0.8,
    },
    content_weight=1.0,
    style_weight=5e5,
    tv_weight=2e-5,
    n_steps=500,
    optimizer="lbfgs",
    init="content",
)

STYLE_PRESETS: Dict[str, StylePreset] = {
    "impressionism": IMPRESSIONISM,
    "cubism":        CUBISM,
    "pointillism":   POINTILLISM,
    "expressionism": EXPRESSIONISM,
    "watercolour":   WATERCOLOUR,
}


def get_preset(name: str) -> StylePreset:
    """Return a preset by name (case-insensitive). Raises KeyError if not found."""
    key = name.lower().strip()
    if key not in STYLE_PRESETS:
        available = ", ".join(STYLE_PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return STYLE_PRESETS[key]
