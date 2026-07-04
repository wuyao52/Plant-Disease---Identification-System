from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.dataset_validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a custom ImageFolder dataset.")
    parser.add_argument("dataset", help="Dataset root directory.")
    parser.add_argument("--min-images-per-class", type=int, default=5)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(
        args.dataset,
        min_images_per_class=args.min_images_per_class,
        check_images=not args.skip_image_check,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
