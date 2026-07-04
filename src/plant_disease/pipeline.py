from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ProjectConfig
from .data import create_dataloaders, ensure_project_dirs, has_imagefolder_data, summarize_imagefolder
from .labels import save_class_names
from .models import count_trainable_parameters, create_model, select_device
from .training import fit, set_seed, write_json


@dataclass(slots=True)
class TrainingRun:
    result: dict[str, object]
    dataset_summary: dict[str, object]
    class_names: list[str]


def build_overrides(
    data_dir: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    checkpoint: str | None = None,
    architecture: str | None = None,
    output_dir: str | None = None,
    pretrained: bool | None = None,
    freeze_backbone: bool | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if data_dir:
        overrides.setdefault("data", {})["root"] = data_dir
    if batch_size:
        overrides.setdefault("data", {})["batch_size"] = batch_size
    if epochs:
        overrides.setdefault("training", {})["epochs"] = epochs
    if checkpoint:
        overrides.setdefault("checkpoint", {})["path"] = checkpoint
    if architecture:
        overrides.setdefault("model", {})["architecture"] = architecture
    if pretrained is not None:
        overrides.setdefault("model", {})["pretrained"] = pretrained
    if freeze_backbone is not None:
        overrides.setdefault("model", {})["freeze_backbone"] = freeze_backbone
    if output_dir:
        overrides.setdefault("outputs", {})["dir"] = output_dir
    return overrides


def train_from_config(
    config: ProjectConfig,
    allow_auto_download: bool = True,
    auto_download_images_per_class: int = 8,
) -> TrainingRun:
    ensure_project_dirs(config.root_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if not has_imagefolder_data(config.data_root):
        if not allow_auto_download:
            raise FileNotFoundError(
                f"Dataset directory is empty or missing: {config.data_root}\n"
                "Use ImageFolder format: <dataset>/<class_name>/<image_file>, or saved split format: "
                "<dataset>/{train,val,test}/<class_name>/<image_file>."
            )
        print(f"dataset_missing={config.data_root}")
        print("downloading a real PlantVillage subset for first-run training...")
        from scripts.download_plantvillage_subset import download_subset

        download_subset(config.data_root, images_per_class=auto_download_images_per_class)

    set_seed(config.seed)
    device = select_device(config.device)
    dataset_summary = summarize_imagefolder(config.data_root)
    dataloaders, class_names = create_dataloaders(config, device)
    model_cfg = config.section("model")
    model = create_model(
        num_classes=len(class_names),
        architecture=str(model_cfg["architecture"]),
        pretrained=bool(model_cfg["pretrained"]),
        dropout=float(model_cfg["dropout"]),
        freeze_backbone=bool(model_cfg["freeze_backbone"]),
    )

    print(f"device={device}")
    print(f"dataset_images={dataset_summary['total_images']} classes={dataset_summary['num_classes']}")
    print(f"classes={len(class_names)}")
    print(f"trainable_parameters={count_trainable_parameters(model):,}")
    result = fit(model, dataloaders, config, device, class_names)

    save_class_names(class_names, config.root_dir / "assets" / "class_names.json")
    write_json(config.output_dir / "dataset_summary.json", dataset_summary)
    write_json(config.output_dir / "history.json", result["history"])
    write_json(config.output_dir / "metrics.json", result)
    print(f"saved_checkpoint={result['checkpoint']}")
    print(f"saved_outputs={config.output_dir}")
    return TrainingRun(result=result, dataset_summary=dataset_summary, class_names=class_names)
