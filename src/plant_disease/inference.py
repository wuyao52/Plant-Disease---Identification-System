from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import ProjectConfig, load_config
from .data import build_transforms
from .labels import display_label
from .models import create_model, find_last_conv2d, select_device


@dataclass(slots=True)
class Prediction:
    class_name: str
    display_name: str
    probability: float
    index: int


class DiseasePredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path = "config.yaml",
        device_name: str | None = None,
    ) -> None:
        self.config = load_config(config_path)
        checkpoint = _load_checkpoint(checkpoint_path)
        self.class_names: list[str] = list(checkpoint["class_names"])
        self.metrics: dict[str, Any] = dict(checkpoint.get("metrics", {}))

        model_cfg = checkpoint.get("config", self.config.values).get("model", self.config.section("model"))
        data_cfg = checkpoint.get("config", self.config.values).get("data", self.config.section("data"))
        self.model_config = dict(model_cfg)
        runtime_device = device_name or self.config.device
        self.device = select_device(runtime_device)
        self.image_size = int(data_cfg["image_size"])
        self.transform = build_transforms(self.image_size, train=False)

        self.model = create_model(
            num_classes=len(self.class_names),
            architecture=str(model_cfg["architecture"]),
            pretrained=False,
            dropout=float(model_cfg.get("dropout", 0.2)),
            freeze_backbone=False,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image, top_k: int = 5) -> list[Prediction]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
        top_k = min(top_k, len(self.class_names))
        values, indices = torch.topk(probabilities, k=top_k)
        return [
            Prediction(
                class_name=self.class_names[int(index)],
                display_name=display_label(self.class_names[int(index)]),
                probability=float(value),
                index=int(index),
            )
            for value, index in zip(values.cpu(), indices.cpu(), strict=True)
        ]

    def gradcam(self, image: Image.Image, class_index: int | None = None) -> Image.Image:
        target_layer = find_last_conv2d(self.model)
        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            activations.append(output.detach())

        def backward_hook(
            _: torch.nn.Module,
            __: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            gradients.append(grad_output[0].detach())

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            if class_index is None:
                class_index = int(logits.argmax(dim=1).item())
            score = logits[0, class_index]
            self.model.zero_grad(set_to_none=True)
            score.backward()

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks did not capture activations")

            weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations[-1]).sum(dim=1).relu()
            cam = torch.nn.functional.interpolate(
                cam.unsqueeze(1),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam_array = cam.detach().cpu().numpy()
            return overlay_heatmap(image, cam_array, self.image_size)
        finally:
            forward_handle.remove()
            backward_handle.remove()


def overlay_heatmap(image: Image.Image, cam_array: np.ndarray, image_size: int) -> Image.Image:
    base = image.convert("RGB").resize((image_size, image_size))
    alpha = Image.fromarray(np.uint8(np.clip(cam_array, 0, 1) * 150), mode="L")
    red_layer = Image.new("RGBA", base.size, (255, 32, 32, 0))
    red_layer.putalpha(alpha)
    return Image.alpha_composite(base.convert("RGBA"), red_layer).convert("RGB")


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "class_names" not in checkpoint:
        raise ValueError("Invalid checkpoint format")
    return checkpoint
