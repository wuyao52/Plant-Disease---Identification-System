from __future__ import annotations

from scripts.create_demo_dataset import CLASSES, generate_demo_dataset


def test_generate_demo_dataset(tmp_path) -> None:
    output = tmp_path / "demo"
    generate_demo_dataset(output, images_per_class=2, size=64, seed=1)
    assert sorted(path.name for path in output.iterdir()) == sorted(CLASSES)
    assert sum(1 for _ in output.rglob("*.jpg")) == len(CLASSES) * 2
