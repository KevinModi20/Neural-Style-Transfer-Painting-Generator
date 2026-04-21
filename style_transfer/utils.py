import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple, Optional


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_normalize   = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
_denormalize = T.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1.0 / s for s in IMAGENET_STD],
)


def load_image(
    path: Union[str, Path],
    size: Optional[Union[int, Tuple[int, int]]] = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Load an image from disk and return a normalized [1, C, H, W] tensor."""
    img = Image.open(path).convert("RGB")

    transforms = []
    if size is not None:
        if isinstance(size, int):
            transforms.append(T.Resize(size, antialias=True))
        else:
            transforms.append(T.Resize(size, antialias=True))
    transforms += [T.ToTensor(), _normalize]

    tensor = T.Compose(transforms)(img).unsqueeze(0)
    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, C, H, W] or [C, H, W] normalized tensor to a PIL Image."""
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)
    t = _denormalize(t).clamp(0, 1)
    return TF.to_pil_image(t)


def save_image(tensor: torch.Tensor, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tensor_to_image(tensor).save(path)


def gram_matrix(feature: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix of a feature map [B, C, H, W]."""
    b, c, h, w = feature.size()
    f = feature.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g.div(c * h * w)


def image_grid(
    images: list,
    titles: Optional[list] = None,
    cols: int = 3,
    figsize_per: Tuple[int, int] = (4, 4),
) -> plt.Figure:
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rows))
    axes = np.array(axes).flatten()

    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = tensor_to_image(img)
        axes[i].imshow(img)
        axes[i].axis("off")
        if titles and i < len(titles):
            axes[i].set_title(titles[i])

    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize src spatial dims to match ref."""
    return torch.nn.functional.interpolate(
        src, size=ref.shape[-2:], mode="bilinear", align_corners=False
    )


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation of a [B, C, H, W] tensor."""
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()
    return diff_h + diff_w
