from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import ProjectConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_datasets(config: ProjectConfig) -> tuple[dict[str, Subset], list[str]]:
    data_cfg = config.section("data")
    data_root = config.data_root
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_root}\n"
            "可先运行：python scripts/download_plantvillage_subset.py --output data/raw/plantvillage"
        )

    if is_saved_split_dataset(data_root):
        image_size = int(data_cfg["image_size"])
        train_dataset = datasets.ImageFolder(data_root / "train", transform=build_transforms(image_size, True))
        val_dataset = datasets.ImageFolder(data_root / "val", transform=build_transforms(image_size, False))
        test_dataset = datasets.ImageFolder(data_root / "test", transform=build_transforms(image_size, False))
        if train_dataset.classes != val_dataset.classes or train_dataset.classes != test_dataset.classes:
            raise ValueError("Saved split dataset must use identical class folders in train/val/test")
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}, list(train_dataset.classes)

    train_dataset = datasets.ImageFolder(data_root, transform=build_transforms(data_cfg["image_size"], True))
    eval_dataset = datasets.ImageFolder(data_root, transform=build_transforms(data_cfg["image_size"], False))
    if len(train_dataset.classes) < 2:
        raise ValueError("Dataset must contain at least two class folders")

    train_indices, val_indices, test_indices = split_indices_by_class(
        targets=train_dataset.targets,
        val_split=float(data_cfg["val_split"]),
        test_split=float(data_cfg["test_split"]),
        seed=config.seed,
    )

    datasets_by_split = {
        "train": Subset(train_dataset, train_indices),
        "val": Subset(eval_dataset, val_indices),
        "test": Subset(eval_dataset, test_indices),
    }
    return datasets_by_split, list(train_dataset.classes)


def split_indices(
    dataset_size: int,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if dataset_size < 3:
        raise ValueError("Dataset must contain at least three images")
    if val_split < 0 or test_split < 0 or val_split + test_split >= 0.8:
        raise ValueError("val_split and test_split must be non-negative and sum to less than 0.8")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    test_size = max(1, int(dataset_size * test_split))
    val_size = max(1, int(dataset_size * val_split))
    train_size = dataset_size - val_size - test_size
    if train_size < 1:
        raise ValueError("Split leaves no training samples")

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]
    return train_indices, val_indices, test_indices


def split_indices_by_class(
    targets: list[int],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not targets:
        raise ValueError("Dataset is empty")
    if val_split < 0 or test_split < 0 or val_split + test_split >= 0.8:
        raise ValueError("val_split and test_split must be non-negative and sum to less than 0.8")

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[int(target)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_index, indices in grouped.items():
        rng.shuffle(indices)
        class_size = len(indices)
        if class_size < 3:
            raise ValueError(
                f"Class index {class_index} has only {class_size} images; at least 3 are required "
                "for train/validation/test split."
            )

        test_size = max(1, int(round(class_size * test_split))) if test_split > 0 else 0
        val_size = max(1, int(round(class_size * val_split))) if val_split > 0 else 0
        while class_size - test_size - val_size < 1:
            if test_size >= val_size and test_size > 0:
                test_size -= 1
            elif val_size > 0:
                val_size -= 1
            else:
                break

        test_indices.extend(indices[:test_size])
        val_indices.extend(indices[test_size : test_size + val_size])
        train_indices.extend(indices[test_size + val_size :])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def create_dataloaders(config: ProjectConfig, device: torch.device) -> tuple[dict[str, DataLoader], list[str]]:
    datasets_by_split, class_names = create_datasets(config)
    data_cfg = config.section("data")
    pin_memory = device.type == "cuda"

    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(data_cfg["num_workers"]),
            pin_memory=pin_memory,
        )
        for split, dataset in datasets_by_split.items()
    }
    return dataloaders, class_names


def ensure_project_dirs(root_dir: str | Path) -> None:
    root = Path(root_dir)
    for relative in ("models", "outputs", "assets", "data/raw", "data/processed"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def has_imagefolder_data(root: str | Path) -> bool:
    root_path = Path(root)
    if not root_path.exists():
        return False
    if is_saved_split_dataset(root_path):
        return True
    return any(
        path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        for path in root_path.glob("*/*")
    )


def summarize_imagefolder(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    if is_saved_split_dataset(root_path):
        split_counts: dict[str, dict[str, int]] = {}
        total_images = 0
        classes: set[str] = set()
        for split in ("train", "val", "test"):
            split_counts[split] = {}
            for class_dir in sorted(path for path in (root_path / split).iterdir() if path.is_dir()):
                count = sum(
                    1
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                )
                split_counts[split][class_dir.name] = count
                classes.add(class_dir.name)
                total_images += count
        return {
            "root": str(root_path),
            "format": "saved_split",
            "num_classes": len(classes),
            "total_images": total_images,
            "split_counts": split_counts,
        }

    class_counts: Counter[str] = Counter()
    total_images = 0
    if root_path.exists():
        for class_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            count = sum(
                1
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            if count:
                class_counts[class_dir.name] = count
                total_images += count
    return {
        "root": str(root_path),
        "format": "imagefolder",
        "num_classes": len(class_counts),
        "total_images": total_images,
        "class_counts": dict(class_counts),
    }


def is_saved_split_dataset(root: str | Path) -> bool:
    root_path = Path(root)
    return all((root_path / split).is_dir() for split in ("train", "val", "test")) and any(
        path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        for split in ("train", "val", "test")
        for path in (root_path / split).glob("*/*")
    )
