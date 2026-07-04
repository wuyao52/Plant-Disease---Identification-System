from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from plant_disease.inference import DiseasePredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict plant disease from one leaf image.")
    parser.add_argument("image", help="Path to image file.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", default="models/plant_disease_mobilenet_v3_small.pt")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    config = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    image = Image.open(args.image)
    predictor = DiseasePredictor(checkpoint, config)
    predictions = predictor.predict(image, top_k=args.top_k)
    for rank, item in enumerate(predictions, start=1):
        print(f"{rank}. {item.display_name}: {item.probability:.4f}")


if __name__ == "__main__":
    main()
