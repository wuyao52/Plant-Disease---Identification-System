from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.config import load_config
from plant_disease.pipeline import build_overrides, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a plant disease recognition model.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--data-dir", default=None, help="Override dataset directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint output path.")
    parser.add_argument("--output-dir", default=None, help="Override directory for metrics and reports.")
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic download when the default dataset directory is empty.",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        choices=("mobilenet_v3_small", "resnet18", "efficientnet_b0"),
        help="Override model architecture.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path, root_dir=ROOT)

    overrides = build_overrides(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        architecture=args.architecture,
        output_dir=args.output_dir,
        pretrained=False if args.no_pretrained else None,
        freeze_backbone=True if args.freeze_backbone else None,
    )
    if overrides:
        config = config.with_overrides(**overrides)

    train_from_config(config, allow_auto_download=not args.no_auto_download)


if __name__ == "__main__":
    main()
