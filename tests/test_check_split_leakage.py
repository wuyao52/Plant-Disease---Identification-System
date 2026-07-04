from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.check_split_leakage import check_leakage


def create_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def test_check_leakage_reports_exact_cross_split_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    create_image(root / "train" / "A" / "a.jpg", (1, 2, 3))
    create_image(root / "test" / "A" / "b.jpg", (1, 2, 3))
    report = check_leakage(root)
    assert report["exact_duplicate_count"] == 1
