from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.prepare_split_dataset import copy_split_dataset


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (60, 140, 70)).save(path)


def test_copy_split_dataset_saves_train_val_test(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    for class_name in ("A", "B"):
        for index in range(10):
            create_image(source / class_name / f"{index}.jpg")

    output = tmp_path / "processed"
    manifest = copy_split_dataset(source, output, train_ratio=0.7, val_ratio=0.1, seed=1)

    assert (output / "split_manifest.json").exists()
    assert sum(manifest["splits"]["train"].values()) == 14  # type: ignore[index, union-attr]
    assert sum(manifest["splits"]["val"].values()) == 2  # type: ignore[index, union-attr]
    assert sum(manifest["splits"]["test"].values()) == 4  # type: ignore[index, union-attr]
    assert (output / "train" / "A").exists()
    assert (output / "test" / "B").exists()
