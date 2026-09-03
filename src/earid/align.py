"""Ear landmark alignment stage.

Trains a 55-landmark regressor on iBUG Collection A ("in-the-wild" ear
images with .pts annotations) and uses it to normalize ear crops:
rotate so the lobe-to-helix axis is vertical, then tightly crop around
the predicted landmarks. Aligned corpora reduce the pose/scale variance
that dominates in-the-wild identification error.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_LANDMARKS = 55
LOBE_POINTS = slice(14, 20)  # ear lobe landmarks
ASCENDING_HELIX_POINTS = slice(0, 4)  # top of the ear

_NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def parse_pts(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = lines.index("{") + 1
    points = [tuple(float(v) for v in line.split()) for line in lines[start : start + NUM_LANDMARKS]]
    if len(points) != NUM_LANDMARKS:
        raise ValueError(f"{path} has {len(points)} points, expected {NUM_LANDMARKS}")
    return np.asarray(points, dtype=np.float32)


@dataclass
class AlignTrainConfig:
    source: str
    output_dir: str
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 8
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 42


class LandmarkDataset(Dataset):
    """Collection A images cropped around the landmark bbox with jitter."""

    def __init__(self, root: Path, image_size: int, train: bool) -> None:
        self.samples = sorted(
            p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.with_suffix(".pts").exists()
        )
        if not self.samples:
            raise ValueError(f"No annotated images found in {root}")
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        image = Image.open(path).convert("RGB")
        points = parse_pts(path.with_suffix(".pts")).copy()

        if self.train:
            angle = random.uniform(-25.0, 25.0)
            image, points = _rotate(image, points, angle)
            margin = random.uniform(0.10, 0.45)
            jitter = 0.06
        else:
            margin = 0.25
            jitter = 0.0
        image, points = _crop_around_landmarks(image, points, margin, jitter)

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            points[:, 0] = image.width - 1 - points[:, 0]

        width, height = image.size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        normalized = points / np.array([width, height], dtype=np.float32)
        tensor = _NORMALIZE(transforms.functional.to_tensor(image))
        return tensor, torch.from_numpy(normalized.reshape(-1))


def _rotate(image: Image.Image, points: np.ndarray, angle_deg: float):
    center = np.array([image.width / 2, image.height / 2], dtype=np.float32)
    rotated = image.rotate(angle_deg, resample=Image.BILINEAR, center=tuple(center), expand=True)
    theta = math.radians(-angle_deg)  # PIL rotates counter-clockwise; points transform inversely
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    shift = np.array([(rotated.width - image.width) / 2, (rotated.height - image.height) / 2], dtype=np.float32)
    return rotated, (points - center) @ rot.T + center + shift


def _crop_around_landmarks(image: Image.Image, points: np.ndarray, margin: float, jitter: float):
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    size = max_xy - min_xy
    pad = size * margin
    if jitter:
        pad = pad + size * np.random.uniform(-jitter, jitter, size=2)
    left, top = np.maximum(min_xy - pad, 0.0)
    right = min(max_xy[0] + pad[0], image.width)
    bottom = min(max_xy[1] + pad[1], image.height)
    image = image.crop((int(left), int(top), int(math.ceil(right)), int(math.ceil(bottom))))
    return image, points - np.array([int(left), int(top)], dtype=np.float32)


def build_landmark_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, NUM_LANDMARKS * 2), nn.Sigmoid())
    return model


def _nme(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean landmark error normalized by the ground-truth bbox diagonal."""
    pred = pred.view(-1, NUM_LANDMARKS, 2)
    target = target.view(-1, NUM_LANDMARKS, 2)
    diag = (target.max(dim=1).values - target.min(dim=1).values).norm(dim=1).clamp(min=1e-6)
    return ((pred - target).norm(dim=2).mean(dim=1) / diag).mean().item()


def train_landmarks(config: AlignTrainConfig) -> dict[str, float]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    root = Path(config.source)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    train_ds = LandmarkDataset(root / "train", config.image_size, train=True)
    test_ds = LandmarkDataset(root / "test", config.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_landmark_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss(beta=0.02)

    best_nme = math.inf
    best_metrics: dict[str, float] = {}
    bad_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for images, targets in tqdm(train_loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            loss = criterion(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        nmes, val_losses = [], []
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                val_losses.append(criterion(preds, targets).item())
                nmes.append(_nme(preds, targets))
        val_nme = float(np.mean(nmes))
        scheduler.step(val_nme)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": float(np.mean(val_losses)),
                    "val_nme": val_nme,
                }
            ),
            flush=True,
        )
        if val_nme < best_nme:
            best_nme = val_nme
            best_metrics = {"epoch": float(epoch), "val_nme": val_nme}
            bad_epochs = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": asdict(config), "best_metrics": best_metrics},
                output_dir / "landmarks.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    (output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    return best_metrics


def load_landmark_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_landmark_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    image_size = int(checkpoint["config"].get("image_size", 128))
    return model, image_size


def predict_landmarks(model: nn.Module, image: Image.Image, image_size: int, device: torch.device) -> np.ndarray:
    tensor = _NORMALIZE(
        transforms.functional.to_tensor(image.resize((image_size, image_size), Image.BILINEAR))
    ).unsqueeze(0)
    with torch.no_grad():
        normalized = model(tensor.to(device)).cpu().numpy().reshape(NUM_LANDMARKS, 2)
    return normalized * np.array([image.width, image.height], dtype=np.float32)


def align_image(
    model: nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
    margin: float = 0.22,
    output_size: int = 224,
) -> Image.Image:
    """Rotate so the lobe-to-helix axis points up, then tightly crop."""
    points = predict_landmarks(model, image, image_size, device)
    lobe = points[LOBE_POINTS].mean(axis=0)
    helix_top = points[ASCENDING_HELIX_POINTS].mean(axis=0)
    axis = helix_top - lobe
    angle = math.degrees(math.atan2(axis[0], -axis[1]))  # 0 when axis points straight up
    rotated, rotated_points = _rotate(image, points, -angle)
    cropped, _ = _crop_around_landmarks(rotated, rotated_points, margin, jitter=0.0)
    if cropped.width < 8 or cropped.height < 8:
        cropped = image  # fall back to the original crop on degenerate predictions
    return cropped.resize((output_size, output_size), Image.BILINEAR)


def align_corpus(
    checkpoint_path: Path,
    source_root: Path,
    output_root: Path,
    device_name: str = "cpu",
    output_size: int = 224,
) -> dict[str, int]:
    device = torch.device(device_name)
    model, image_size = load_landmark_model(checkpoint_path, device)
    written = failed = 0
    files = [p for p in sorted(source_root.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    for path in tqdm(files):
        target = output_root / path.relative_to(source_root).with_suffix(".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(path) as img:
                aligned = align_image(model, img.convert("RGB"), image_size, device, output_size=output_size)
            aligned.save(target, "JPEG", quality=95)
            written += 1
        except Exception:  # noqa: BLE001 - skip unreadable files, keep aligning
            failed += 1
    return {"written": written, "failed": failed}
