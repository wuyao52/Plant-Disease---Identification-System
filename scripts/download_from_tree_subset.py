from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests


RAW_BASE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"
DEFAULT_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_tree_paths(tree_file: str | Path) -> list[str]:
    payload = json.loads(Path(tree_file).read_text(encoding="utf-8-sig"))
    paths: list[str] = []
    for item in payload.get("tree", []):
        path = item.get("path", "")
        if item.get("type") == "blob" and Path(path).suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
    return paths


def select_class_paths(
    all_paths: list[str],
    classes: list[str],
    images_per_class: int,
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for class_name in classes:
        candidates = [path for path in all_paths if path.startswith(class_name + "/")]
        if len(candidates) < images_per_class:
            raise ValueError(
                f"Class {class_name} only has {len(candidates)} cached paths; "
                f"cannot select {images_per_class} images."
            )
        rng.shuffle(candidates)
        selected[class_name] = candidates[:images_per_class]
    return selected


def raw_url(path: str) -> str:
    return f"{RAW_BASE}/{quote(path, safe='/()_,.-')}"


def download_one(path: str, destination_root: Path, timeout: int = 25, retries: int = 3) -> tuple[str, bool, str]:
    class_name, file_name = path.split("/", 1)
    destination = destination_root / class_name / file_name
    if destination.exists() and destination.stat().st_size > 0:
        return path, True, "exists"

    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(raw_url(path), timeout=timeout)
            response.raise_for_status()
            destination.write_bytes(response.content)
            return path, True, "downloaded"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.8 * attempt)
    return path, False, last_error


def download_subset_from_tree(
    tree_file: str | Path,
    output: str | Path,
    classes: list[str] | None = None,
    images_per_class: int = 80,
    seed: int = 42,
    workers: int = 8,
) -> dict[str, object]:
    output_path = Path(output)
    selected_classes = classes or DEFAULT_CLASSES
    all_paths = load_tree_paths(tree_file)
    selected = select_class_paths(all_paths, selected_classes, images_per_class, seed)
    selected_paths = [path for paths in selected.values() for path in paths]

    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(download_one, path, output_path): path
            for path in selected_paths
        }
        completed = 0
        for future in as_completed(future_map):
            path, ok, message = future.result()
            completed += 1
            if not ok:
                failures.append({"path": path, "error": message})
            if completed % 25 == 0 or completed == len(selected_paths):
                print(f"downloaded_or_checked={completed}/{len(selected_paths)} failures={len(failures)}")

    counts = {}
    for class_name in selected_classes:
        class_dir = output_path / class_name
        counts[class_name] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    manifest = {
        "source": RAW_BASE,
        "tree_file": str(Path(tree_file).resolve()),
        "output": str(output_path.resolve()),
        "classes": selected_classes,
        "images_per_class": images_per_class,
        "seed": seed,
        "counts": counts,
        "failures": failures,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; see dataset_manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a balanced PlantVillage subset from a cached tree.")
    parser.add_argument("--tree-file", default="outputs/plantvillage_color_tree.json")
    parser.add_argument("--output", default="data/raw/plantvillage_subset")
    parser.add_argument("--images-per-class", type=int, default=80)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = download_subset_from_tree(
        tree_file=args.tree_file,
        output=args.output,
        classes=args.classes,
        images_per_class=args.images_per_class,
        seed=args.seed,
        workers=args.workers,
    )
    print("download complete")
    for class_name, count in manifest["counts"].items():  # type: ignore[union-attr]
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
