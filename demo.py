"""
Programmatic demo of all three style transfer modes.

Creates a synthetic 256×256 content image and a synthetic style image,
runs all three pipelines, and saves results to outputs/.

No GPU required; runs on CPU for verification.
"""

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

from style_transfer.neural_style import NeuralStyleTransfer, NSTConfig
from style_transfer.fast_style import FastStyleNetwork
from style_transfer.gan import StyleGAN, GANConfig
from style_transfer.losses import PerceptualLoss
from style_transfer.presets import get_preset
from style_transfer.utils import load_image, save_image, tensor_to_image, image_grid


OUTPUT_DIR = Path("outputs/demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
print(f"Running demo on: {DEVICE}")


# ---------------------------------------------------------------------------
# Synthetic images (no need for real artwork files)
# ---------------------------------------------------------------------------

def make_content_image(size: int = 256) -> torch.Tensor:
    """Generate a simple geometric pattern as content."""
    img = Image.new("RGB", (size, size), (200, 220, 240))
    draw = ImageDraw.Draw(img)
    # Draw concentric rectangles
    for i in range(0, size // 2, 20):
        c = int(180 - i * 0.8)
        draw.rectangle([i, i, size - i, size - i], outline=(c, c // 2, 255 - c), width=2)
    # Draw diagonal lines
    for j in range(0, size, 30):
        draw.line([(0, j), (j, 0)], fill=(100, 160, 200), width=1)

    t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return t(img).unsqueeze(0).to(DEVICE)


def make_style_image(size: int = 256) -> torch.Tensor:
    """Generate a colourful swirl as a style reference."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            angle  = np.arctan2(dy, dx)
            radius = np.sqrt(dx ** 2 + dy ** 2)
            r = int(128 + 127 * np.sin(angle * 3 + radius / 20))
            g = int(128 + 127 * np.cos(angle * 5 - radius / 15))
            b = int(128 + 127 * np.sin(radius / 12))
            arr[y, x] = [r, g, b]

    img = Image.fromarray(arr)
    t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return t(img).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Demo runs
# ---------------------------------------------------------------------------

def demo_nst(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    print("\n--- Optimization-based NST (50 steps, quick demo) ---")
    preset = get_preset("impressionism")
    config = NSTConfig(
        content_layers=preset.content_layers,
        style_layers=preset.style_layers,
        content_weights=preset.content_weights,
        style_weights=preset.style_weights,
        content_weight=preset.content_weight,
        style_weight=preset.style_weight,
        tv_weight=preset.tv_weight,
        n_steps=50,           # short for demo
        optimizer="adam",     # faster than lbfgs for demo
        lr=0.02,
        init="content",
    )
    nst = NeuralStyleTransfer(device=DEVICE)
    result = nst.run(content, style, config)
    save_image(result, OUTPUT_DIR / "nst_impressionism.png")
    print(f"  Saved: {OUTPUT_DIR / 'nst_impressionism.png'}")
    return result


def demo_fast(content: torch.Tensor) -> torch.Tensor:
    print("\n--- Fast Feed-Forward (random weights, untrained) ---")
    net = FastStyleNetwork(device=DEVICE)
    result = net.stylize(content)
    save_image(result, OUTPUT_DIR / "fast_untrained.png")
    print(f"  Saved: {OUTPUT_DIR / 'fast_untrained.png'}")
    return result


def demo_gan(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    print("\n--- GAN Generator (random weights, untrained) ---")
    gan = StyleGAN(style_image=style, device=DEVICE)
    result = gan.generate(content)
    save_image(result, OUTPUT_DIR / "gan_untrained.png")
    print(f"  Saved: {OUTPUT_DIR / 'gan_untrained.png'}")
    return result


def demo_presets(content: torch.Tensor, style: torch.Tensor) -> None:
    """Run NST with each preset for 30 steps to preview style character."""
    print("\n--- Preset gallery (30 steps each) ---")
    from style_transfer.presets import STYLE_PRESETS
    results, titles = [], []
    for name, preset in STYLE_PRESETS.items():
        config = NSTConfig(
            content_layers=preset.content_layers,
            style_layers=preset.style_layers,
            style_weights=preset.style_weights,
            content_weight=preset.content_weight,
            style_weight=preset.style_weight,
            tv_weight=preset.tv_weight,
            n_steps=30,
            optimizer="adam",
            lr=0.02,
        )
        nst = NeuralStyleTransfer(device=DEVICE)
        r = nst.run(content, style, config)
        results.append(r)
        titles.append(name)
        save_image(r, OUTPUT_DIR / f"preset_{name}.png")
        print(f"  [{name}] saved")

    fig = image_grid(results, titles=titles, cols=3)
    fig.savefig(OUTPUT_DIR / "preset_gallery.png", dpi=100, bbox_inches="tight")
    print(f"  Gallery saved: {OUTPUT_DIR / 'preset_gallery.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    content = make_content_image(256)
    style   = make_style_image(256)

    # Save raw inputs for reference
    save_image(content, OUTPUT_DIR / "input_content.png")
    save_image(style,   OUTPUT_DIR / "input_style.png")

    nst_result  = demo_nst(content, style)
    fast_result = demo_fast(content)
    gan_result  = demo_gan(content, style)
    demo_presets(content, style)

    # Final comparison grid
    all_imgs   = [content, style, nst_result, fast_result, gan_result]
    all_titles = ["Content", "Style", "NST", "Fast (untrained)", "GAN (untrained)"]
    fig = image_grid(all_imgs, titles=all_titles, cols=3)
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=120, bbox_inches="tight")
    print(f"\nComparison grid saved: {OUTPUT_DIR / 'comparison.png'}")
    print("\nDemo complete.")
