from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def list_class_images(source: Path) -> dict[str, list[Path]]:
    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    class_images: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        images = sorted(
            path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            class_images[class_dir.name] = images
    if len(class_images) < 2:
        raise ValueError("Source dataset must contain at least two non-empty class folders")
    return class_images


def split_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_per_class: int | None,
) -> dict[str, list[Path]]:
    selected = list(images)
    rng = random.Random(seed)
    rng.shuffle(selected)
    if max_per_class is not None:
        selected = selected[:max_per_class]
    if len(selected) < 5:
        raise ValueError("Each class needs at least 5 images to create a stable train/val/test split")

    train_count = max(1, int(round(len(selected) * train_ratio)))
    val_count = max(1, int(round(len(selected) * val_ratio)))
    while train_count + val_count >= len(selected):
        if val_count > 1:
            val_count -= 1
        else:
            train_count -= 1

    return {
        "train": selected[:train_count],
        "val": selected[train_count : train_count + val_count],
        "test": selected[train_count + val_count :],
    }


def copy_split_dataset(
    source: str | Path,
    output: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_per_class: int | None = None,
    force: bool = False,
) -> dict[str, object]:
    source_path = Path(source)
    output_path = Path(output)
    if force and output_path.exists():
        shutil.rmtree(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_path}. Use --force.")
    output_path.mkdir(parents=True, exist_ok=True)

    class_images = list_class_images(source_path)
    manifest: dict[str, object] = {
        "source": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1 - train_ratio - val_ratio,
        "max_per_class": max_per_class,
        "classes": sorted(class_images),
        "splits": {split: {} for split in SPLITS},
        "files": {split: {} for split in SPLITS},
    }

    for class_index, (class_name, images) in enumerate(sorted(class_images.items())):
        split_map = split_images(images, train_ratio, val_ratio, seed + class_index, max_per_class)
        for split, split_files in split_map.items():
            destination_dir = output_path / split / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            copied_names: list[str] = []
            for source_file in split_files:
                destination = destination_dir / source_file.name
                shutil.copy2(source_file, destination)
                copied_names.append(source_file.name)
            manifest["splits"][split][class_name] = len(copied_names)  # type: ignore[index]
            manifest["files"][split][class_name] = copied_names  # type: ignore[index]

    manifest_path = output_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed train/val/test dataset copy.")
    parser.add_argument("--source", default="data/raw/plantvillage")
    parser.add_argument("--output", default="data/processed/plantvillage_split")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = copy_split_dataset(
        source=args.source,
        output=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_per_class=args.max_per_class,
        force=args.force,
    )
    print(f"saved_split={manifest['output']}")
    for split in SPLITS:
        total = sum(manifest["splits"][split].values())  # type: ignore[index, union-attr]
        print(f"{split}: {total}")


if __name__ == "__main__":
    main()
