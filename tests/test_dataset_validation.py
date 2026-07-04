from __future__ import annotations

from pathlib import Path

from PIL import Image

from plant_disease.dataset_validation import validate_dataset


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (80, 140, 80)).save(path)


def test_validate_imagefolder_dataset(tmp_path: Path) -> None:
    for class_name in ("A", "B"):
        for index in range(5):
            create_image(tmp_path / class_name / f"{index}.jpg")

    report = validate_dataset(tmp_path)
    assert report["valid"] is True
    assert report["format"] == "imagefolder"
    assert report["num_classes"] == 2
    assert report["total_images"] == 10


def test_validate_dataset_reports_invalid_image(tmp_path: Path) -> None:
    create_image(tmp_path / "A" / "0.jpg")
    create_image(tmp_path / "B" / "0.jpg")
    (tmp_path / "B" / "bad.jpg").write_text("not an image", encoding="utf-8")

    report = validate_dataset(tmp_path, min_images_per_class=1)
    assert report["valid"] is False
    assert report["invalid_images"]
