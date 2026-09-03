from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class PaperEarCNN(nn.Module):
    """A compact six-layer CNN inspired by the PMC7594944 ear-recognition paper."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def build_model(backbone: str, num_classes: int, pretrained: bool = True, loss: str = "ce") -> nn.Module:
    if loss == "arcface":
        return ArcFaceModel(backbone, num_classes, pretrained=pretrained)
    backbone = backbone.lower()
    if backbone == "paper_cnn":
        return PaperEarCNN(num_classes)
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported backbone: {backbone}")


EMBEDDING_DIMS = {"resnet18": 512, "efficientnet_b0": 1280, "paper_cnn": 256}


class ArcFaceModel(nn.Module):
    """Backbone + additive angular margin (ArcFace) classification head.

    Training requires targets so the margin can be applied to the true
    class; inference without targets returns scaled cosine logits.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        scale: float = 30.0,
        margin: float = 0.30,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone.lower()
        self.backbone = build_model(self.backbone_name, num_classes, pretrained=pretrained, loss="ce")
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, EMBEDDING_DIMS[self.backbone_name]))
        nn.init.xavier_uniform_(self.weight)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return extract_embeddings(self.backbone, self.backbone_name, x)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(self.embed(x), dim=1), F.normalize(self.weight, dim=1))
        if targets is None:
            return cosine * self.scale
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_mask = F.one_hot(targets, num_classes=cosine.size(1)).bool()
        margined = torch.cos(theta + self.margin)
        return torch.where(target_mask, margined, cosine) * self.scale


def extract_embeddings(model: nn.Module, backbone: str, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, ArcFaceModel):
        return model.embed(x)
    backbone = backbone.lower()
    if backbone == "paper_cnn":
        if not isinstance(model, PaperEarCNN):
            raise TypeError("paper_cnn embeddings require a PaperEarCNN model")
        return model.forward_features(x)
    if backbone == "resnet18":
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)
    if backbone == "efficientnet_b0":
        x = model.features(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)
    raise ValueError(f"Unsupported backbone: {backbone}")


def load_matching_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    current_state = model.state_dict()
    if isinstance(model, ArcFaceModel) and not any(key.startswith("backbone.") for key in state_dict):
        # Allow warm-starting an ArcFace model from a plain classifier checkpoint.
        state_dict = {f"backbone.{key}": value for key, value in state_dict.items()}
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return list(missing), list(unexpected)
