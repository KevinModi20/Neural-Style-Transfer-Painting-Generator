"""
Training script for the fast feed-forward style transfer network (Johnson et al.).

Trains a TransformNet using a perceptual loss (VGG content + Gram-matrix style
+ total variation) on an arbitrary image dataset.  One network per style.

Usage
-----
  python train.py \\
      --style  assets/styles/starry_night.jpg \\
      --data   /path/to/coco/train2017 \\
      --output checkpoints/starry_night.pth \\
      --preset impressionism \\
      --epochs 2 --batch 4 --size 256
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from style_transfer.fast_style import FastStyleNetwork
from style_transfer.losses import PerceptualLoss
from style_transfer.presets import get_preset, STYLE_PRESETS
from style_transfer.utils import load_image, save_image, tensor_to_image


def build_dataloader(data_dir: str, size: int, batch: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(size, antialias=True),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load style image
    style_img = load_image(args.style, size=args.size, device=device)

    # Resolve per-layer weights from preset (if given)
    preset = get_preset(args.preset) if args.preset else None

    fsn = FastStyleNetwork(device=device)

    perc = PerceptualLoss(
        style_image=style_img,
        content_weight=preset.content_weight if preset else 1.0,
        style_weight=preset.style_weight   if preset else 1e5,
        tv_weight=preset.tv_weight      if preset else 1e-6,
        style_weights=preset.style_weights   if preset else None,
        device=device,
    )

    optimizer = optim.Adam(fsn.module.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    dataloader = build_dataloader(args.data, args.size, args.batch)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        fsn.module.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            generated = fsn.module(images)
            loss_dict = perc(generated, images)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fsn.module.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                style=f"{loss_dict['style'].item():.4f}",
            )

            # Save a sample every 500 batches
            if i % 500 == 0:
                sample_path = Path(args.output).parent / f"sample_e{epoch}_b{i}.png"
                save_image(generated[0:1], sample_path)

        scheduler.step()
        avg = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} — avg loss: {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            fsn.save(args.output)
            print(f"  Saved best checkpoint → {args.output}")

    print("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fast style-transfer network")
    parser.add_argument("--style",   required=True, help="Path to style image")
    parser.add_argument("--data",    required=True, help="ImageFolder-style dataset root")
    parser.add_argument("--output",  default="checkpoints/model.pth", help="Checkpoint path")
    parser.add_argument("--preset",  default=None, choices=list(STYLE_PRESETS.keys()),
                        help="Artistic style preset for loss weights")
    parser.add_argument("--epochs",  type=int,   default=2)
    parser.add_argument("--batch",   type=int,   default=4)
    parser.add_argument("--size",    type=int,   default=256)
    parser.add_argument("--lr",      type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
