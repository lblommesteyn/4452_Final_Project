from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _background(size: int, rng: np.random.Generator) -> Image.Image:
    array = rng.normal(loc=18.0, scale=9.0, size=(size, size, 3))
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(rng.integers(12, 28)):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        radius = int(rng.integers(1, 3))
        brightness = int(rng.integers(120, 220))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(brightness, brightness, brightness))
    return image


def _make_lens_image(size: int, rng: np.random.Generator) -> Image.Image:
    image = _background(size, rng)
    draw = ImageDraw.Draw(image)
    cx = int(rng.integers(size // 3, (2 * size) // 3))
    cy = int(rng.integers(size // 3, (2 * size) // 3))
    radius = int(rng.integers(size // 7, size // 4))
    thickness = int(rng.integers(3, 7))

    ring_color = tuple(int(v) for v in rng.integers(150, 240, size=3))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=ring_color, width=thickness)

    core_radius = max(4, radius // 3)
    core_color = tuple(int(v) for v in rng.integers(180, 255, size=3))
    draw.ellipse((cx - core_radius, cy - core_radius, cx + core_radius, cy + core_radius), fill=core_color)

    arc_offset = int(rng.integers(-6, 6))
    arc_bbox = (cx - radius - 5, cy - radius + arc_offset, cx + radius + 5, cy + radius + arc_offset)
    draw.arc(arc_bbox, start=15, end=160, fill=(255, 220, 120), width=max(2, thickness - 1))
    draw.arc(arc_bbox, start=200, end=320, fill=(120, 200, 255), width=max(2, thickness - 1))

    return image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 1.4))))


def _make_non_lens_image(size: int, rng: np.random.Generator) -> Image.Image:
    image = _background(size, rng)
    draw = ImageDraw.Draw(image)
    cx = int(rng.integers(size // 4, (3 * size) // 4))
    cy = int(rng.integers(size // 4, (3 * size) // 4))
    rx = int(rng.integers(size // 8, size // 3))
    ry = int(rng.integers(size // 10, size // 4))
    angle_offset = int(rng.integers(-10, 10))

    galaxy_color = tuple(int(v) for v in rng.integers(100, 210, size=3))
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=galaxy_color)
    draw.arc(
        (cx - rx - 4, cy - ry - 4, cx + rx + 4, cy + ry + 4),
        start=40 + angle_offset,
        end=110 + angle_offset,
        fill=(90, 120, 180),
        width=2,
    )
    for _ in range(rng.integers(2, 5)):
        x1 = int(rng.integers(0, size))
        y1 = int(rng.integers(0, size))
        x2 = int(rng.integers(0, size))
        y2 = int(rng.integers(0, size))
        draw.line((x1, y1, x2, y2), fill=(80, 80, 120), width=int(rng.integers(1, 3)))

    return image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.1))))


def create_synthetic_dataset(
    output_dir: str | Path,
    image_size: int = 128,
    train_per_class: int = 48,
    val_per_class: int = 16,
    test_per_class: int = 16,
    seed: int = 42,
) -> Path:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    plan = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }

    rows: list[dict[str, str | int]] = []
    for split, count in plan.items():
        for label_name, label_value, generator in (
            ("lens", 1, _make_lens_image),
            ("non_lens", 0, _make_non_lens_image),
        ):
            split_dir = images_dir / split / label_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                image = generator(image_size, rng)
                file_name = f"{label_name}_{index:04d}.png"
                image_path = split_dir / file_name
                image.save(image_path)
                rows.append(
                    {
                        "image_path": str(image_path.relative_to(output_dir)),
                        "label": label_value,
                        "split": split,
                        "source": "synthetic",
                    }
                )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "split", "source"])
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path

