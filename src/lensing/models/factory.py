from __future__ import annotations

from torch import nn
from torchvision import models

from lensing.config import ModelConfig


def _maybe_freeze_backbone(model: nn.Module, trainable_names: set[str]) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name.startswith(trainable_name) for trainable_name in trainable_names)


def create_model(config: ModelConfig) -> nn.Module:
    if config.name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if config.pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features, 1),
        )
        if config.freeze_backbone:
            _maybe_freeze_backbone(model, {"fc"})
        return model

    if config.name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if config.pretrained else None
        model = models.vit_b_16(weights=weights, image_size=config.image_size)
        hidden_dim = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(hidden_dim, 1),
        )
        if config.freeze_backbone:
            _maybe_freeze_backbone(model, {"heads"})
        return model

    raise ValueError(f"Unsupported model name: {config.name}")

