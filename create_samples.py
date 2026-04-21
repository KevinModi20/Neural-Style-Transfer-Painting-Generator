"""
Generate synthetic sample images into assets/ so stylize.py and demo.py
can be run immediately without downloading any external artwork.

  python create_samples.py
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"  Created: {path}")


# ---------------------------------------------------------------------------
# Content image — architectural-style scene with depth layers
# ---------------------------------------------------------------------------

def make_content(size: int = 512) -> Image.Image:
    img = Image.new("RGB", (size, size), (135, 175, 210))
    draw = ImageDraw.Draw(img)

    # Sky gradient
    for y in range(size // 2):
        t = y / (size // 2)
        r = int(100 + 80 * t)
        g = int(150 + 60 * t)
        b = int(220 - 30 * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))

    # Ground
    for y in range(size // 2, size):
        t = (y - size // 2) / (size // 2)
        r = int(80 + 60 * t)
        g = int(120 + 30 * t)
        b = int(60 + 20 * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))

    # Buildings
    buildings = [
        (50,  200, 150, size - 80,  (180, 160, 140)),
        (170, 150, 290, size - 80,  (160, 140, 120)),
        (310, 250, 390, size - 80,  (200, 180, 160)),
        (400, 120, 480, size - 80,  (170, 150, 130)),
    ]
    for x0, y0, x1, y1, col in buildings:
        draw.rectangle([x0, y0, x1, y1], fill=col, outline=(80, 70, 60), width=2)
        # Windows
        for wx in range(x0 + 10, x1 - 10, 20):
            for wy in range(y0 + 15, y1 - 10, 25):
                light = (255, 240, 180) if (wx + wy) % 3 == 0 else (60, 70, 90)
                draw.rectangle([wx, wy, wx + 10, wy + 15], fill=light)

    # Tree silhouettes
    for tx in [80, 240, 450]:
        draw.rectangle([tx - 4, size - 130, tx + 4, size - 60], fill=(40, 60, 30))
        draw.ellipse([tx - 28, size - 160, tx + 28, size - 100], fill=(50, 100, 40))

    return img.filter(ImageFilter.SMOOTH_MORE)


# ---------------------------------------------------------------------------
# Style images — one per artistic movement
# ---------------------------------------------------------------------------

def make_impressionism(size: int = 256) -> Image.Image:
    """Loose colour dabs in warm/cool clusters."""
    rng = np.random.default_rng(0)
    arr = np.zeros((size, size, 3), dtype=np.float32)

    palettes = [
        [(255, 200, 80),  (255, 160, 40)],   # warm yellows / oranges
        [(80,  160, 220), (120, 200, 240)],   # cool blues
        [(180, 220, 120), (120, 180, 60)],    # greens
        [(240, 120, 120), (200, 80,  80)],    # reds
    ]
    for _ in range(6000):
        x  = rng.integers(0, size)
        y  = rng.integers(0, size)
        r  = rng.integers(4, 16)
        pal = palettes[rng.integers(len(palettes))]
        col = np.array(pal[rng.integers(2)], dtype=np.float32) / 255.0
        # Soft gaussian dab
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    w = np.exp(-(dx * dx + dy * dy) / (r * r))
                    arr[ny, nx] = np.clip(arr[ny, nx] * (1 - w) + col * w, 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img.filter(ImageFilter.SMOOTH)


def make_cubism(size: int = 256) -> Image.Image:
    """Fragmented geometric planes in earthy tones."""
    img = Image.new("RGB", (size, size), (200, 180, 140))
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(1)

    colours = [
        (180, 140, 90), (140, 100, 60), (220, 190, 140),
        (90,  80,  70), (160, 120, 80), (100, 130, 160),
        (80,  100, 80), (200, 160, 100),
    ]
    for _ in range(80):
        n = rng.integers(4, 8)
        cx = rng.integers(0, size)
        cy = rng.integers(0, size)
        r  = rng.integers(20, 80)
        angles = sorted(rng.uniform(0, 2 * np.pi, n))
        pts = [(int(cx + r * np.cos(a)), int(cy + r * np.sin(a))) for a in angles]
        col = colours[rng.integers(len(colours))]
        draw.polygon(pts, fill=col, outline=(40, 30, 20))

    return img


def make_pointillism(size: int = 256) -> Image.Image:
    """Dense pure-colour dots."""
    img = Image.new("RGB", (size, size), (240, 235, 220))
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(2)

    dot_colours = [
        (220, 60,  40),  (240, 180, 20), (50,  120, 200),
        (60,  160, 80),  (180, 60, 180), (240, 120, 40),
        (40,  180, 200), (200, 200, 60),
    ]
    for _ in range(12000):
        x   = rng.integers(0, size)
        y   = rng.integers(0, size)
        r   = rng.integers(2, 5)
        col = dot_colours[rng.integers(len(dot_colours))]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    return img


def make_expressionism(size: int = 256) -> Image.Image:
    """Van Gogh-style swirling strokes."""
    arr = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy = size / 2, size / 2
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            angle  = np.arctan2(dy, dx)
            radius = np.sqrt(dx * dx + dy * dy)
            swirl  = angle + radius / 30
            r = 0.45 + 0.45 * np.sin(swirl * 2.5)
            g = 0.35 + 0.35 * np.cos(swirl * 3.0 + 1)
            b = 0.25 + 0.45 * np.sin(swirl * 1.8 + 2)
            arr[y, x] = [r, g, b]

    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return img.filter(ImageFilter.SMOOTH_MORE)


def make_watercolour(size: int = 256) -> Image.Image:
    """Soft translucent washes."""
    rng = np.random.default_rng(4)
    arr = np.ones((size, size, 3), dtype=np.float32) * 0.96  # off-white paper

    wash_colours = [
        (0.6, 0.8, 0.9),  # pale blue
        (0.9, 0.7, 0.6),  # salmon
        (0.7, 0.9, 0.7),  # mint
        (0.95, 0.9, 0.6), # straw yellow
        (0.8, 0.7, 0.9),  # lavender
    ]
    for _ in range(20):
        cx  = rng.integers(0, size)
        cy  = rng.integers(0, size)
        rx  = rng.integers(40, 120)
        ry  = rng.integers(30, 100)
        col = np.array(wash_colours[rng.integers(len(wash_colours))], dtype=np.float32)
        alpha = rng.uniform(0.08, 0.25)
        for py in range(max(0, cy - ry), min(size, cy + ry)):
            for px in range(max(0, cx - rx), min(size, cx + rx)):
                ex = ((px - cx) / rx) ** 2
                ey = ((py - cy) / ry) ** 2
                if ex + ey <= 1.0:
                    w = alpha * (1 - (ex + ey)) * rng.uniform(0.5, 1.0)
                    arr[py, px] = np.clip(arr[py, px] * (1 - w) + col * w, 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(radius=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating sample images...")

    save(make_content(512),       Path("assets/content/photo.jpg"))
    save(make_impressionism(256), Path("assets/styles/impressionism.jpg"))
    save(make_cubism(256),        Path("assets/styles/cubism.jpg"))
    save(make_pointillism(256),   Path("assets/styles/pointillism.jpg"))
    save(make_expressionism(256), Path("assets/styles/expressionism.jpg"))
    save(make_watercolour(256),   Path("assets/styles/watercolour.jpg"))

    print("\nDone. You can now run:")
    print("  python stylize.py nst --content assets/content/photo.jpg \\")
    print("                        --style   assets/styles/impressionism.jpg \\")
    print("                        --preset  impressionism \\")
    print("                        --output  outputs/result.png")
    print("\n  python demo.py")
