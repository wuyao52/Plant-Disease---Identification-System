from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any


API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
DEFAULT_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]
USER_AGENT = "codex-plant-disease-project"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def request_api(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def request_json(url: str) -> list[dict[str, Any]]:
    data = request_api(url)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected GitHub API response from {url}: {data}")
    return data


def list_available_classes() -> list[str]:
    entries = request_json(f"{API_BASE}?ref=master")
    return sorted(entry["name"] for entry in entries if entry.get("type") == "dir")


def list_class_images(class_name: str) -> list[dict[str, str]]:
    entries = request_json(f"{API_BASE}/{class_name}?ref=master")
    images = []
    for entry in entries:
        name = entry.get("name", "")
        git_url = entry.get("git_url")
        if entry.get("type") == "file" and git_url and Path(name).suffix.lower() in IMAGE_SUFFIXES:
            images.append({"name": name, "git_url": git_url})
    return sorted(images, key=lambda item: item["name"])


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                destination.write_bytes(response.read())
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)


def download_blob_file(git_url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            data = request_api(git_url)
            content = data.get("content")
            encoding = data.get("encoding")
            if not isinstance(content, str) or encoding != "base64":
                raise RuntimeError(f"Unexpected blob response for {git_url}")
            destination.write_bytes(base64.b64decode(content))
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt)


def download_subset(
    output: str | Path,
    classes: list[str] | None = None,
    images_per_class: int = 40,
    seed: int = 42,
    force: bool = False,
) -> dict[str, int]:
    output_path = Path(output)
    selected_classes = classes or DEFAULT_CLASSES
    if force and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for class_name in selected_classes:
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = list(class_dir.glob("*"))
        if len(existing) >= images_per_class:
            counts[class_name] = len([path for path in existing if path.suffix.lower() in IMAGE_SUFFIXES])
            print(f"{class_name}: already has {counts[class_name]} images")
            continue

        images = list_class_images(class_name)
        if not images:
            raise RuntimeError(f"No images found for class: {class_name}")
        rng.shuffle(images)
        selected = images[:images_per_class]
        for index, item in enumerate(selected, start=1):
            destination = class_dir / item["name"]
            if destination.exists():
                continue
            print(f"downloading {class_name} {index}/{len(selected)}: {item['name']}")
            download_blob_file(item["git_url"], destination)
        counts[class_name] = sum(
            1 for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    manifest = {
        "source": "https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color",
        "classes": selected_classes,
        "images_per_class": images_per_class,
        "counts": counts,
    }
    (output_path / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a real PlantVillage image subset from GitHub.")
    parser.add_argument("--output", default="data/raw/plantvillage")
    parser.add_argument("--images-per-class", type=int, default=8)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-classes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_classes:
        for class_name in list_available_classes():
            print(class_name)
        return

    counts = download_subset(
        output=args.output,
        classes=args.classes,
        images_per_class=args.images_per_class,
        seed=args.seed,
        force=args.force,
    )
    print("download complete")
    for class_name, count in counts.items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
