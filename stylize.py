"""
CLI entry point for neural style transfer.

Three modes
-----------
  nst    Optimization-based (Gatys et al.) — slow but high quality, no training.
  fast   Feed-forward network (Johnson et al.) — requires a trained checkpoint.
  gan    U-Net + PatchGAN enhancer — requires a trained checkpoint.

Examples
--------
  # Optimization-based with a named preset
  python stylize.py nst \\
      --content assets/content/photo.jpg \\
      --style   assets/styles/starry_night.jpg \\
      --preset  impressionism \\
      --output  outputs/result.png

  # Fast inference with a pre-trained checkpoint
  python stylize.py fast \\
      --content    assets/content/photo.jpg \\
      --checkpoint checkpoints/starry_night.pth \\
      --output     outputs/fast_result.png

  # GAN inference
  python stylize.py gan \\
      --content    assets/content/photo.jpg \\
      --checkpoint checkpoints/gan/epoch_050.pth \\
      --output     outputs/gan_result.png

  # List presets
  python stylize.py presets
"""

import argparse
import sys
from pathlib import Path

import torch

from style_transfer.neural_style import NeuralStyleTransfer, NSTConfig
from style_transfer.fast_style import FastStyleNetwork
from style_transfer.gan import StyleGAN
from style_transfer.presets import STYLE_PRESETS, get_preset
from style_transfer.utils import load_image, save_image


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_nst(args: argparse.Namespace) -> None:
    device = _device()
    print(f"Device: {device}")

    content = load_image(args.content, size=args.size, device=device)
    style   = load_image(args.style,   size=args.size, device=device)

    if args.preset:
        p = get_preset(args.preset)
        config = NSTConfig(
            content_layers=p.content_layers,
            style_layers=p.style_layers,
            content_weights=p.content_weights,
            style_weights=p.style_weights,
            content_weight=p.content_weight,
            style_weight=p.style_weight,
            tv_weight=p.tv_weight,
            n_steps=args.steps or p.n_steps,
            optimizer=p.optimizer,
            init=p.init,
            save_every=args.save_every,
            output_dir=str(Path(args.output).parent),
        )
    else:
        config = NSTConfig(
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            tv_weight=args.tv_weight,
            n_steps=args.steps or 500,
            save_every=args.save_every,
            output_dir=str(Path(args.output).parent),
        )

    nst = NeuralStyleTransfer(device=device)
    result = nst.run(content, style, config)
    save_image(result, args.output)
    print(f"Saved → {args.output}")


def cmd_fast(args: argparse.Namespace) -> None:
    device  = _device()
    content = load_image(args.content, size=args.size, device=device)
    net     = FastStyleNetwork.from_checkpoint(args.checkpoint, device=device)
    result  = net.stylize(content)
    save_image(result, args.output)
    print(f"Saved → {args.output}")


def cmd_gan(args: argparse.Namespace) -> None:
    device  = _device()
    content = load_image(args.content, size=args.size, device=device)
    gan     = StyleGAN.from_checkpoint(args.checkpoint, device=device)
    result  = gan.generate(content)
    save_image(result, args.output)
    print(f"Saved → {args.output}")


def cmd_presets(_args: argparse.Namespace) -> None:
    print("\nAvailable style presets:\n")
    for name, p in STYLE_PRESETS.items():
        print(f"  {name:<16}  {p.description}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stylize",
        description="Neural Style Transfer — painting generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- nst ----
    p_nst = sub.add_parser("nst", help="Optimization-based style transfer (Gatys et al.)")
    p_nst.add_argument("--content",        required=True)
    p_nst.add_argument("--style",          required=True)
    p_nst.add_argument("--output",         default="outputs/result_nst.png")
    p_nst.add_argument("--preset",         choices=list(STYLE_PRESETS.keys()))
    p_nst.add_argument("--size",           type=int,   default=512)
    p_nst.add_argument("--steps",          type=int,   default=0, help="0 = use preset default")
    p_nst.add_argument("--content-weight", type=float, default=1.0)
    p_nst.add_argument("--style-weight",   type=float, default=1e6)
    p_nst.add_argument("--tv-weight",      type=float, default=1e-4)
    p_nst.add_argument("--save-every",     type=int,   default=0)
    p_nst.set_defaults(func=cmd_nst)

    # ---- fast ----
    p_fast = sub.add_parser("fast", help="Fast feed-forward style transfer")
    p_fast.add_argument("--content",    required=True)
    p_fast.add_argument("--checkpoint", required=True)
    p_fast.add_argument("--output",     default="outputs/result_fast.png")
    p_fast.add_argument("--size",       type=int, default=512)
    p_fast.set_defaults(func=cmd_fast)

    # ---- gan ----
    p_gan = sub.add_parser("gan", help="GAN-based style enhancement")
    p_gan.add_argument("--content",    required=True)
    p_gan.add_argument("--checkpoint", required=True)
    p_gan.add_argument("--output",     default="outputs/result_gan.png")
    p_gan.add_argument("--size",       type=int, default=256)
    p_gan.set_defaults(func=cmd_gan)

    # ---- presets ----
    p_pre = sub.add_parser("presets", help="List available style presets")
    p_pre.set_defaults(func=cmd_presets)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
