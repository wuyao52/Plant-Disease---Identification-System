from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "plant-disease-recognition", "seed": 42},
    "data": {
        "root": "data/raw/plantvillage",
        "image_size": 224,
        "batch_size": 32,
        "val_split": 0.15,
        "test_split": 0.10,
        "num_workers": 0,
    },
    "model": {
        "architecture": "mobilenet_v3_small",
        "pretrained": True,
        "dropout": 0.2,
        "freeze_backbone": False,
    },
    "training": {
        "epochs": 10,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "patience": 5,
        "amp": False,
    },
    "checkpoint": {"path": "models/plant_disease_mobilenet_v3_small.pt"},
    "inference": {"top_k": 5},
    "runtime": {"device": "auto"},
    "outputs": {"dir": "outputs"},
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating the input dictionaries."""
    merged = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class ProjectConfig:
    values: dict[str, Any]
    root_dir: Path = field(default_factory=lambda: Path.cwd())

    @property
    def seed(self) -> int:
        return int(self.values["project"]["seed"])

    @property
    def data_root(self) -> Path:
        return resolve_path(self.values["data"]["root"], self.root_dir)

    @property
    def checkpoint_path(self) -> Path:
        return resolve_path(self.values["checkpoint"]["path"], self.root_dir)

    @property
    def output_dir(self) -> Path:
        return resolve_path(self.values["outputs"]["dir"], self.root_dir)

    @property
    def device(self) -> str:
        return str(self.values["runtime"]["device"])

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.values[name])

    def with_overrides(self, **overrides: Any) -> "ProjectConfig":
        merged = deep_update(self.values, overrides)
        return ProjectConfig(merged, self.root_dir)


def resolve_path(path_value: str | Path, root_dir: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(root_dir).resolve() / path


def load_config(config_path: str | Path = "config.yaml", root_dir: str | Path | None = None) -> ProjectConfig:
    path = Path(config_path)
    project_root = Path(root_dir).resolve() if root_dir else path.resolve().parent

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
    else:
        loaded = {}

    values = deep_update(DEFAULT_CONFIG, loaded)
    return ProjectConfig(values=values, root_dir=project_root)


def save_config(config: ProjectConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.values, file, allow_unicode=True, sort_keys=False)
