from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


CLASSES = ("Synthetic_Healthy", "Synthetic_Rust", "Synthetic_Mildew")


def draw_leaf(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> list[tuple[int, int]]:
    cx = width // 2 + rng.randint(-12, 12)
    cy = height // 2 + rng.randint(-8, 8)
    leaf_width = rng.randint(max(18, int(width * 0.34)), max(24, int(width * 0.46)))
    leaf_height = rng.randint(max(24, int(height * 0.58)), max(30, int(height * 0.72)))
    points: list[tuple[int, int]] = []
    for step in range(80):
        theta = 2 * math.pi * step / 80
        radius_x = leaf_width * math.sin(theta)
        radius_y = leaf_height * math.cos(theta)
        taper = 0.55 + 0.45 * abs(math.cos(theta))
        x = int(cx + radius_x * taper * 0.55)
        y = int(cy + radius_y * 0.48)
        points.append((x, y))
    color = (54 + rng.randint(-6, 18), 132 + rng.randint(-10, 24), 55 + rng.randint(-8, 18))
    draw.polygon(points, fill=color)
    draw.line((cx, cy - leaf_height // 2, cx, cy + leaf_height // 2), fill=(36, 97, 42), width=3)
    return points


def add_symptoms(draw: ImageDraw.ImageDraw, label: str, width: int, height: int, rng: random.Random) -> None:
    def point_inside_leaf() -> tuple[int, int]:
        x_margin = max(8, int(width * 0.26))
        y_margin = max(8, int(height * 0.20))
        return (
            rng.randint(x_margin, max(x_margin, width - x_margin)),
            rng.randint(y_margin, max(y_margin, height - y_margin)),
        )

    if label == "Synthetic_Healthy":
        for _ in range(8):
            x, y = point_inside_leaf()
            draw.ellipse((x, y, x + 3, y + 3), fill=(80, 155, 70))
        return

    if label == "Synthetic_Rust":
        for _ in range(28):
            x, y = point_inside_leaf()
            r = rng.randint(max(2, width // 80), max(3, width // 28))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(166, 91, 33))
            draw.ellipse((x - r // 2, y - r // 2, x + r // 2, y + r // 2), fill=(214, 138, 50))
        return

    for _ in range(35):
        x, y = point_inside_leaf()
        r = rng.randint(max(2, width // 90), max(3, width // 32))
        shade = rng.randint(210, 245)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(shade, shade, shade))


def generate_image(label: str, size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    background = (226 + rng.randint(-6, 6), 232 + rng.randint(-5, 5), 218 + rng.randint(-6, 6))
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    draw_leaf(draw, size, size, rng)
    add_symptoms(draw, label, size, size, rng)
    return image.filter(ImageFilter.GaussianBlur(radius=0.35))


def generate_demo_dataset(output: str | Path, images_per_class: int = 36, size: int = 224, seed: int = 42) -> None:
    output_path = Path(output)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for class_index, label in enumerate(CLASSES):
        class_dir = output_path / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index in range(images_per_class):
            image = generate_image(label, size=size, seed=seed + class_index * 1000 + image_index)
            image.save(class_dir / f"{label}_{image_index:04d}.jpg", quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic leaf dataset for smoke tests.")
    parser.add_argument("--output", default="data/raw/demo_leaf")
    parser.add_argument("--images-per-class", type=int, default=36)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_demo_dataset(args.output, args.images_per_class, args.size, args.seed)
    print(f"created_demo_dataset={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
