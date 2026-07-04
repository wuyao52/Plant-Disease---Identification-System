from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tv_models


SUPPORTED_ARCHITECTURES = ("mobilenet_v3_small", "resnet18", "efficientnet_b0")


def create_model(
    num_classes: int,
    architecture: str = "mobilenet_v3_small",
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    architecture = architecture.lower()
    if architecture == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            freeze_parameters(model)
        in_features = model.classifier[-1].in_features
        model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        if freeze_backbone:
            freeze_parameters(model)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        return model

    if architecture == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            freeze_parameters(model)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        return model

    supported = ", ".join(SUPPORTED_ARCHITECTURES)
    raise ValueError(f"Unsupported architecture '{architecture}'. Choose one of: {supported}")


def freeze_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def find_last_conv2d(model: nn.Module) -> nn.Conv2d:
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in model")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def select_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
