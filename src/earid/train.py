from __future__ import annotations

import copy
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from .data import (
    EarDataset,
    build_manifest_digest,
    discover_samples,
    limit_samples_per_identity,
    prepare_dataset,
    save_manifest,
    split_samples,
)
from .metrics import classification_metrics
from .models import ArcFaceModel, build_model, load_matching_state_dict


@dataclass
class TrainConfig:
    source: str
    output_dir: str
    cache_dir: str
    include_full_face: bool
    image_size: int
    val_ratio: float
    test_ratio: float
    seed: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    backbone: str
    pretrained: bool
    patience: int
    num_workers: int
    device: str
    init_checkpoint: str | None = None
    max_samples_per_identity: int | None = None
    loss: str = "ce"


@dataclass
class MonteCarloResult:
    run_index: int
    seed: int
    output_dir: str
    metrics: dict[str, float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=(-4, 4)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                transforms.RandomErasing(p=0.18, scale=(0.02, 0.08), ratio=(0.3, 3.0)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    truth = targets.detach().cpu().numpy()
    return classification_metrics(truth, preds)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    all_logits = []
    all_targets = []

    for images, targets, _ in tqdm(loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                if training and isinstance(model, ArcFaceModel):
                    logits = model(images, targets)
                else:
                    logits = model(images)
                loss = criterion(logits, targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(loss.detach().item())
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(logits, targets)
    metrics["loss"] = float(sum(losses) / max(1, len(losses)))
    return metrics


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    label_to_index: dict[str, int],
    config: TrainConfig,
    best_metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "config": asdict(config),
            "best_metrics": best_metrics,
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_initial_weights(model: nn.Module, checkpoint_path: str | None) -> dict[str, list[str]] | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = load_matching_state_dict(model, checkpoint["model_state_dict"])
    return {"missing": missing, "unexpected": unexpected}


def load_samples(config: TrainConfig):
    source_path = Path(config.source)
    cache_dir = Path(config.cache_dir)
    prepared_root = prepare_dataset(source_path, cache_dir)
    samples = discover_samples(prepared_root, include_full_face=config.include_full_face)
    samples = limit_samples_per_identity(
        samples, config.max_samples_per_identity, config.seed
    )
    if not samples:
        raise ValueError(f"No images found in {prepared_root}")
    label_to_index = {label: idx for idx, label in enumerate(sorted({s.label for s in samples}))}
    return prepared_root, samples, label_to_index


def train_single_run(config: TrainConfig) -> dict[str, float]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    _, samples, label_to_index = load_samples(config)
    splits = split_samples(samples, config.val_ratio, config.test_ratio, config.seed)
    for split_name in ("train", "val", "test"):
        if not splits[split_name]:
            raise ValueError(
                f"{split_name} split is empty. Individual-identification "
                "training requires identities with at least three images. "
                "Two-image morphology datasets such as BabyEar4k cannot "
                "provide a valid same-person train/validation/test protocol."
            )

    save_manifest(samples, output_dir / "manifest.csv")

    train_ds = EarDataset(splits["train"], transform=build_transforms(config.image_size, True))
    val_ds = EarDataset(splits["val"], transform=build_transforms(config.image_size, False))
    test_ds = EarDataset(splits["test"], transform=build_transforms(config.image_size, False))

    # Balance datasets first and identities second so a large source corpus
    # cannot dominate smaller cohorts even when classes are frequency-balanced.
    class_counts: dict[int, int] = {}
    class_domains: dict[int, str] = {}
    for s in splits["train"]:
        class_counts[s.label_index] = class_counts.get(s.label_index, 0) + 1
        class_domains[s.label_index] = s.label.split("/", 1)[0]
    domain_class_counts = Counter(class_domains.values())
    sample_weights = [
        1.0
        / (
            domain_class_counts[class_domains[s.label_index]]
            * class_counts[s.label_index]
        )
        for s in splits["train"]
    ]
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device(config.device)
    model = build_model(
        config.backbone, num_classes=len(label_to_index), pretrained=config.pretrained, loss=config.loss
    ).to(device)
    init_info = load_initial_weights(model, config.init_checkpoint)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = -math.inf
    best_metrics: dict[str, float] = {}
    best_state: dict | None = None
    bad_epochs = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            # Persist the best checkpoint immediately so an interrupted run
            # never loses completed training progress.
            torch.save(
                {
                    "model_state_dict": best_state,
                    "label_to_index": label_to_index,
                    "config": asdict(config),
                    "best_metrics": best_metrics,
                },
                output_dir / "checkpoint_best_val.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, criterion, None, device, None)
    best_metrics.update(
        {
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "dataset_digest": build_manifest_digest(samples),
            "num_samples": float(len(samples)),
            "num_classes": float(len(label_to_index)),
        }
    )
    if init_info is not None:
        best_metrics["init_missing_layers"] = float(len(init_info["missing"]))
        best_metrics["init_unexpected_layers"] = float(len(init_info["unexpected"]))
    save_checkpoint(output_dir, model, label_to_index, config, best_metrics)
    return best_metrics


def _aggregate_run_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {"runs": float(len(results))}
    for key in ("val_accuracy", "test_accuracy", "val_macro_f1", "test_macro_f1", "train_accuracy"):
        values = [result[key] for result in results if key in result]
        if values:
            summary[f"{key}_mean"] = float(statistics.mean(values))
            summary[f"{key}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    best = max(results, key=lambda r: r.get("val_accuracy", float("-inf")))
    summary["best_val_accuracy"] = float(best.get("val_accuracy", 0.0))
    summary["best_test_accuracy"] = float(best.get("test_accuracy", 0.0))
    return summary


def train(config: TrainConfig) -> dict[str, float]:
    return train_single_run(config)


def finetune(config: TrainConfig) -> dict[str, float]:
    if config.init_checkpoint is None:
        raise ValueError("init_checkpoint is required for finetune")
    return train_single_run(config)


def train_monte_carlo(config: TrainConfig, runs: int, seed_step: int = 1) -> dict[str, float]:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    root_output_dir = Path(config.output_dir)
    _, samples, _ = load_samples(config)
    save_manifest(samples, root_output_dir / "manifest.csv")

    results: list[dict[str, float]] = []
    run_records: list[MonteCarloResult] = []
    for run_index in range(runs):
        run_seed = config.seed + run_index * seed_step
        run_output_dir = root_output_dir / f"run-{run_index + 1:02d}-seed-{run_seed}"
        run_config = replace(config, seed=run_seed, output_dir=str(run_output_dir))
        metrics = train_single_run(run_config)
        results.append(metrics)
        run_records.append(
            MonteCarloResult(
                run_index=run_index + 1,
                seed=run_seed,
                output_dir=str(run_output_dir),
                metrics=metrics,
            )
        )

    summary = _aggregate_run_metrics(results)
    summary["seed"] = float(config.seed)
    summary["seed_step"] = float(seed_step)
    summary["runs_requested"] = float(runs)
    (root_output_dir / "monte_carlo_summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "runs": [asdict(record) for record in run_records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
