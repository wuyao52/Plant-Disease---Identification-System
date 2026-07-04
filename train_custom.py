from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.config import load_config
from plant_disease.data import is_saved_split_dataset
from plant_disease.dataset_validation import validate_dataset
from plant_disease.pipeline import build_overrides, train_from_config
from scripts.check_split_leakage import check_leakage
from scripts.prepare_split_dataset import copy_split_dataset


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._") or "custom_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with a user's own plant disease dataset.")
    parser.add_argument("--source", required=True, help="Custom dataset root.")
    parser.add_argument("--name", default=None, help="Run name used for processed data, model, and reports.")
    parser.add_argument("--config", default="config.yaml", help="Base config file.")
    parser.add_argument("--prepared-dir", default=None, help="Where to save generated train/val/test split.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--architecture",
        choices=("mobilenet_v3_small", "resnet18", "efficientnet_b0"),
        default=None,
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite generated prepared dataset.")
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    run_name = slugify(args.name or source.name)
    prepared_dir = Path(args.prepared_dir or ROOT / "data" / "processed" / f"custom_{run_name}")
    output_dir = args.output_dir or f"outputs/runs/{run_name}"
    output_path = Path(output_dir)
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    checkpoint = args.checkpoint or f"models/{run_name}.pt"

    validation = validate_dataset(source, check_images=not args.skip_image_check)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["valid"]:
        raise SystemExit("Custom dataset validation failed. Fix the errors above and retry.")

    if is_saved_split_dataset(source):
        training_data_dir = source
        print(f"using_existing_split={training_data_dir}")
    else:
        manifest = copy_split_dataset(
            source=source,
            output=prepared_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed or 42,
            max_per_class=args.max_per_class,
            force=args.force,
        )
        training_data_dir = Path(str(manifest["output"]))
        print(f"prepared_split={training_data_dir}")

    leakage_report = check_leakage(training_data_dir)
    leakage_path = output_path / "leakage_report.json"
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_path.write_text(json.dumps(leakage_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if leakage_report["exact_duplicate_count"] or leakage_report["near_duplicate_count"]:
        print("warning=Cross-split duplicate or near-duplicate images were found. See leakage_report.json.")

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path, root_dir=ROOT)
    overrides = build_overrides(
        data_dir=str(training_data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint=checkpoint,
        architecture=args.architecture,
        output_dir=output_dir,
        pretrained=False if args.no_pretrained else None,
        freeze_backbone=True if args.freeze_backbone else None,
    )
    if args.seed is not None:
        overrides.setdefault("project", {})["seed"] = args.seed
    train_from_config(config.with_overrides(**overrides), allow_auto_download=False)


if __name__ == "__main__":
    main()
