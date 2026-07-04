from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(root: Path) -> list[tuple[str, str, Path]]:
    items: list[tuple[str, str, Path]] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                items.append((split, path.parent.name, path))
    return items


def md5(path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((hash_size + 1, hash_size))
        pixels = list(image.getdata())

    value = 0
    for row in range(hash_size):
        start = row * (hash_size + 1)
        for col in range(hash_size):
            value = (value << 1) | int(pixels[start + col] > pixels[start + col + 1])
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def check_leakage(root: str | Path, near_threshold: int = 4) -> dict[str, object]:
    root_path = Path(root)
    items = image_files(root_path)
    split_counts = {split: 0 for split in ("train", "val", "test")}
    name_sets = {split: set() for split in ("train", "val", "test")}
    md5_seen: dict[str, tuple[str, str]] = {}
    exact_duplicates: list[dict[str, str]] = []

    hashes: list[tuple[str, str, str, int]] = []
    for split, class_name, path in items:
        split_counts[split] += 1
        name_sets[split].add(path.name)
        checksum = md5(path)
        if checksum in md5_seen and md5_seen[checksum][0] != split:
            exact_duplicates.append(
                {
                    "file": str(path),
                    "split": split,
                    "duplicate_of": md5_seen[checksum][1],
                    "duplicate_split": md5_seen[checksum][0],
                }
            )
        else:
            md5_seen[checksum] = (split, str(path))
        hashes.append((split, class_name, str(path), dhash(path)))

    same_name_overlap = {
        "train_val": sorted(name_sets["train"] & name_sets["val"]),
        "train_test": sorted(name_sets["train"] & name_sets["test"]),
        "val_test": sorted(name_sets["val"] & name_sets["test"]),
    }

    near_duplicates: list[dict[str, object]] = []
    for index, left in enumerate(hashes):
        for right in hashes[index + 1 :]:
            if left[0] == right[0]:
                continue
            distance = hamming(left[3], right[3])
            if distance <= near_threshold:
                near_duplicates.append(
                    {
                        "distance": distance,
                        "left_split": left[0],
                        "left_class": left[1],
                        "left_file": left[2],
                        "right_split": right[0],
                        "right_class": right[1],
                        "right_file": right[2],
                    }
                )

    return {
        "root": str(root_path.resolve()),
        "near_threshold": near_threshold,
        "split_counts": split_counts,
        "same_name_overlap_counts": {key: len(value) for key, value in same_name_overlap.items()},
        "exact_duplicate_count": len(exact_duplicates),
        "near_duplicate_count": len(near_duplicates),
        "same_name_overlap": same_name_overlap,
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates[:50],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exact and near-duplicate leakage across splits.")
    parser.add_argument("--root", default="data/processed/plantvillage_split")
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--output", default="outputs/leakage_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_leakage(args.root, args.near_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
