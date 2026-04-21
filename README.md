# Neural Style Transfer — Painting Generator

Neural Style Transfer system that applies artistic styles (Impressionism, Cubism, Pointillism) to photographs using VGG19 CNNs and GANs. Implements optimization-based NST (Gatys et al.), fast feed-forward networks (Johnson et al.), and U-Net + PatchGAN enhancement with perceptual loss functions.

Generate paintings from photographs by applying artistic styles using deep learning.
Supports three methods: optimization-based NST, fast feed-forward networks, and GAN-based enhancement.

---

## Examples

```
Your Photo  +  Style Reference  →  Stylized Painting
```

| Content | Style | Output |
|---------|-------|--------|
| photo.jpg | impressionism.jpg | Monet-like soft brushstrokes |
| photo.jpg | cubism.jpg | Fragmented geometric planes |
| photo.jpg | pointillism.jpg | Dense pure-colour dots |
| photo.jpg | expressionism.jpg | Van Gogh-style swirls |
| photo.jpg | watercolour.jpg | Soft translucent washes |

---

## How It Works

A pretrained VGG19 CNN separates **what** is in an image (content) from **how** it looks (style/texture). The system exploits this by optimizing a generated image to simultaneously match:

- The **content** of your photo — object shapes, structure (deep VGG layers)
- The **style** of the artwork — textures, colors, brushstrokes (Gram matrices of shallow layers)

---

## Project Structure

```
project/
├── requirements.txt          # Python dependencies
├── create_samples.py         # Generate sample content + style images
├── stylize.py                # CLI entry point
├── train.py                  # Train fast style-transfer network
├── demo.py                   # Full demo with synthetic images
├── assets/
│   ├── content/              # Put your photos here
│   └── styles/               # Put your style/artwork images here
├── outputs/                  # Generated results saved here
└── style_transfer/
    ├── utils.py              # Image I/O, Gram matrix, visualization
    ├── vgg.py                # Frozen VGG19 feature extractor
    ├── losses.py             # Content, Style, TV, Perceptual losses
    ├── neural_style.py       # Optimization-based NST (Gatys et al.)
    ├── fast_style.py         # Feed-forward TransformNet (Johnson et al.)
    ├── gan.py                # U-Net Generator + PatchGAN Discriminator
    └── presets.py            # Artistic style presets
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1 — generate sample images (only needed once)
python create_samples.py

# Step 2 — run style transfer
python stylize.py nst \
    --content assets/content/photo.jpg \
    --style   assets/styles/impressionism.jpg \
    --preset  impressionism \
    --output  outputs/result.png
```

---

## Three Methods

### 1. NST — Optimization-based (Gatys et al., 2015)

Directly optimizes the output image over hundreds of steps. No training required.
Best quality but slowest — takes a few minutes on CPU.

```bash
python stylize.py nst \
    --content assets/content/photo.jpg \
    --style   assets/styles/impressionism.jpg \
    --preset  impressionism \
    --output  outputs/result.png
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | none | Artistic style preset (see presets below) |
| `--size` | 512 | Resize image to this square size before processing |
| `--steps` | preset default | Number of optimization steps (fewer = faster) |
| `--content-weight` | 1.0 | How strongly to preserve photo content |
| `--style-weight` | 1e6 | How strongly to apply artistic style |
| `--tv-weight` | 1e-4 | Smoothness regularization |
| `--save-every` | 0 | Save intermediate result every N steps (0 = off) |

**Reduce heat / speed up:**
```bash
python stylize.py nst \
    --content assets/content/photo.jpg \
    --style   assets/styles/impressionism.jpg \
    --preset  impressionism \
    --output  outputs/result.png \
    --steps 100 --size 256
```

---

### 2. Fast — Feed-forward Network (Johnson et al., 2016)

A ResNet-based network trained once per style. At inference runs in a single forward pass — real-time speed.

**Train:**
```bash
python train.py \
    --style   assets/styles/impressionism.jpg \
    --data    /path/to/image/dataset \
    --output  checkpoints/impressionism.pth \
    --preset  impressionism \
    --epochs  2 \
    --batch   4 \
    --size    256
```

> The `--data` folder must follow ImageFolder format (images inside a subfolder).
> A large dataset like [COCO](https://cocodataset.org) works well.

**Inference:**
```bash
python stylize.py fast \
    --content    assets/content/photo.jpg \
    --checkpoint checkpoints/impressionism.pth \
    --output     outputs/fast_result.png
```

---

### 3. GAN — U-Net + PatchGAN Enhancer

U-Net generator with PatchGAN discriminator trained using perceptual + adversarial loss.
Produces richer high-frequency texture detail.

**Inference (with trained checkpoint):**
```bash
python stylize.py gan \
    --content    assets/content/photo.jpg \
    --checkpoint checkpoints/gan/epoch_050.pth \
    --output     outputs/gan_result.png
```

---

## Style Presets

Presets contain tuned hyperparameters (layer weights, content/style balance) for each artistic movement.

```bash
python stylize.py presets
```

| Preset | Description |
|--------|-------------|
| `impressionism` | Soft brushstrokes, luminous colour mixing (Monet, Renoir) |
| `cubism` | Fragmented geometric planes, angular decomposition (Picasso, Braque) |
| `pointillism` | Pure-colour dot patterns, optical colour mixing (Seurat, Signac) |
| `expressionism` | Bold distorted forms, emotionally charged colour (Van Gogh, Munch) |
| `watercolour` | Translucent washes, soft colour blooms (Turner, Sargent) |

---

## Run Full Demo

Runs all three pipelines using synthetic images. No real photos or internet required.

```bash
python demo.py
```

Results saved to `outputs/demo/`:
- `input_content.png` — synthetic content image
- `input_style.png` — synthetic style image
- `nst_impressionism.png` — NST result
- `fast_untrained.png` — fast network result (random weights)
- `gan_untrained.png` — GAN result (random weights)
- `preset_*.png` — one result per preset
- `comparison.png` — side-by-side grid of all results

---

## Using Your Own Images

Replace the sample images with any photo or artwork:

```bash
# Use your own photo and a painting you downloaded
python stylize.py nst \
    --content assets/content/my_photo.jpg \
    --style   assets/styles/starry_night.jpg \
    --preset  expressionism \
    --output  outputs/my_result.png
```

Any `.jpg` or `.png` file works. The image is auto-resized to `--size` (default 512px).

---

## Method Comparison

| | NST | Fast | GAN |
|---|---|---|---|
| Speed | Slow (minutes) | Real-time | Real-time |
| Quality | Highest | Good | Best texture |
| Training needed | No | Yes (per style) | Yes |
| Best for | One-off high quality results | Batch processing | Rich artistic detail |

---

## Architecture Overview

- **VGG19** — frozen feature extractor with AvgPool (smoother gradients than MaxPool)
- **Gram matrices** — capture style as correlations between feature channels
- **Instance Normalization** — used in TransformNet and U-Net for style-independent processing
- **LSGAN objective** — least-squares loss for stable PatchGAN training
- **Perceptual loss** — combines VGG content loss + Gram style loss + total variation

---

## Results & Findings

### Artistic Transformations
The system successfully maintains **object recognizability** in the output — faces, buildings, and landscapes remain clearly identifiable while their surface appearance is repainted in the target artistic style. This is controlled by the content loss anchoring the generated image to deep VGG features that encode structure rather than texture.

### Style-specific Observations

| Style | Key Observation |
|-------|----------------|
| Impressionism | Low-frequency VGG layers (relu2_1, relu3_1) carry the most style signal — boosting their weight produces richer colour blending |
| Cubism | Uniform weighting across all layers produces the characteristic fragmented look; reducing content weight allows more geometric distortion |
| Pointillism | Concentrating style weight on early layers (relu1_1, relu2_1) captures fine dot patterns better than deeper layers |
| Expressionism | Higher content weight preserves emotional subject clarity while still applying bold colour distortion |
| Watercolour | Low TV weight allows soft natural noise that mimics paper texture bleeding |

### Perceptual Loss Functions
Using a combination of content loss (MSE on VGG features) and style loss (MSE on Gram matrices) across multiple layers produces significantly better results than single-layer optimization:
- **Single layer** — tends to over-stylize or lose content structure
- **Multi-layer** — balances texture at different scales (fine brushstrokes + broad colour regions)
- **Total variation loss** — critical for suppressing high-frequency noise without blurring edges

### GAN vs Optimization-based
- **Optimization-based NST** produces the highest fidelity per image but takes minutes per result
- **GAN-based enhancement** generates richer high-frequency texture detail (sharp brushstroke edges, dot boundaries) that the smooth perceptual loss alone tends to suppress
- **Fast feed-forward** offers the best speed/quality tradeoff for applying one style to many images

### Content vs Style Weight Tradeoff
- Increasing `--style-weight` → stronger artistic effect, objects become harder to recognize
- Increasing `--content-weight` → photo stays recognizable, style effect is subtler
- Recommended starting ratio: `style_weight / content_weight = 1e6` (default)

---

## References

- Gatys et al. (2015) — [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- Johnson et al. (2016) — [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155)
- Ulyanov et al. (2017) — [Instance Normalization](https://arxiv.org/abs/1607.08022)
- Isola et al. (2017) — [Image-to-Image Translation with Conditional GANs](https://arxiv.org/abs/1611.07004)
