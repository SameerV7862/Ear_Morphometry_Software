from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import (
    discover_samples,
    limit_samples_per_identity,
    prepare_dataset,
    split_samples,
)
from .models import build_model, extract_embeddings
from .train import TrainConfig, finetune, train, train_monte_carlo


def _common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="Dataset ZIP or extracted directory")
    parser.add_argument("--output-dir", default="runs/earid", help="Training output directory")
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory for extracted ZIPs")
    parser.add_argument("--include-full-face", action="store_true", help="Keep full-face images in the dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0", "paper_cnn"])
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-samples-per-identity",
        type=int,
        help="Deterministically cap images per person for large datasets",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earid", description="Ear-based human identification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Extract the ZIP and build a manifest")
    prepare_parser.add_argument("--source", required=True)
    prepare_parser.add_argument("--cache-dir", default=".cache")
    prepare_parser.add_argument("--include-full-face", action="store_true")
    prepare_parser.add_argument("--manifest", default="runs/earid/manifest.csv")

    train_parser = subparsers.add_parser("train", help="Train the model")
    _common_train_args(train_parser)

    finetune_parser = subparsers.add_parser("finetune", help="Continue training from an existing checkpoint")
    _common_train_args(finetune_parser)
    finetune_parser.add_argument("--init-checkpoint", required=True, help="Checkpoint to initialize from")
    finetune_parser.set_defaults(pretrained=False)

    mc_parser = subparsers.add_parser("train-mc", help="Train repeated subject-safe runs")
    _common_train_args(mc_parser)
    mc_parser.add_argument("--mc-runs", type=int, default=5, help="Number of repeated runs")
    mc_parser.add_argument("--mc-seed-step", type=int, default=1, help="Seed increment between runs")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint")
    eval_parser.add_argument("--source", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--cache-dir", default=".cache")
    eval_parser.add_argument("--include-full-face", action="store_true")
    eval_parser.add_argument("--image-size", type=int, default=224)
    eval_parser.add_argument("--batch-size", type=int, default=16)
    eval_parser.add_argument("--num-workers", type=int, default=2)
    eval_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    eval_parser.add_argument("--calibration-bins", type=int, default=10)
    eval_parser.add_argument("--output-json", help="Optional path for detailed evaluation metrics")

    predict_parser = subparsers.add_parser("predict", help="Predict identity for one image")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--image-size", type=int, default=224)
    predict_parser.add_argument("--tta", type=int, default=1, help="Number of test-time augmentation passes")
    predict_parser.add_argument("--reject-below", type=float, help="Return no-decision below this confidence")
    predict_parser.add_argument("--temperature", type=float, default=1.0, help="Validation-fitted calibration temperature")
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    cross_parser = subparsers.add_parser("cross-test", help="Cross-dataset identification using embeddings")
    cross_parser.add_argument("--source", required=True, help="Different ear dataset ZIP or directory")
    cross_parser.add_argument("--checkpoint", required=True)
    cross_parser.add_argument("--cache-dir", default=".cache")
    cross_parser.add_argument("--include-full-face", action="store_true")
    cross_parser.add_argument("--image-size", type=int, default=224)
    cross_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    open_set_parser = subparsers.add_parser(
        "open-set-test",
        help="Evaluate enrolled identities against separate unknown identities",
    )
    open_set_parser.add_argument("--known-source", required=True)
    open_set_parser.add_argument("--unknown-validation-source", required=True)
    open_set_parser.add_argument("--unknown-test-source", required=True)
    open_set_parser.add_argument("--checkpoint", required=True)
    open_set_parser.add_argument("--cache-dir", default=".cache")
    open_set_parser.add_argument("--image-size", type=int, default=224)
    open_set_parser.add_argument("--batch-size", type=int, default=16)
    open_set_parser.add_argument("--target-far", type=float, default=0.01)
    open_set_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    open_set_parser.add_argument("--output-json")

    validate_parser = subparsers.add_parser("validate", help="Smoke test the data pipeline and model")
    validate_parser.add_argument("--source", required=True)
    validate_parser.add_argument("--cache-dir", default=".cache")
    validate_parser.add_argument("--include-full-face", action="store_true")
    validate_parser.add_argument("--image-size", type=int, default=224)
    validate_parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0", "paper_cnn"])
    validate_parser.add_argument("--pretrained", action="store_true")
    validate_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return parser


def _build_eval_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _build_tta_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.10, p=0.30),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=(-2, 2)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _embedding_loader(samples, image_size, batch_size=16, num_workers=0):
    dataset = []
    transform = _build_eval_transforms(image_size)
    for sample in samples:
        dataset.append(sample)
    from torch.utils.data import Dataset

    class _EmbeddingDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            sample = self.items[index]
            with Image.open(sample.path) as image:
                return transform(image.convert("RGB")), sample.label, sample.subject_id, str(sample.path)

    return DataLoader(_EmbeddingDataset(dataset), batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _mean_normalize(vectors: torch.Tensor) -> torch.Tensor:
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    return vectors


def _select_open_set_threshold(
    known_scores,
    known_predictions,
    known_labels,
    unknown_scores,
    target_far: float,
) -> float:
    known_scores = np.asarray(known_scores, dtype=np.float64)
    known_predictions = np.asarray(known_predictions)
    known_labels = np.asarray(known_labels)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    if known_scores.size == 0 or unknown_scores.size == 0:
        raise ValueError("Threshold selection requires known and unknown samples")

    candidate_scores = np.unique(np.concatenate([known_scores, unknown_scores]))
    thresholds = np.concatenate(
        [[float("-inf")], np.nextafter(candidate_scores, float("inf"))]
    )
    valid_candidates = []
    for threshold in thresholds:
        false_accept_rate = float((unknown_scores >= threshold).mean())
        if false_accept_rate > target_far:
            continue
        identification_rate = float(
            (
                (known_predictions == known_labels)
                & (known_scores >= threshold)
            ).mean()
        )
        valid_candidates.append(
            (identification_rate, -false_accept_rate, -threshold, threshold)
        )
    if not valid_candidates:
        raise RuntimeError("No threshold satisfies the requested validation FAR")
    return float(max(valid_candidates)[-1])


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        root = prepare_dataset(Path(args.source), Path(args.cache_dir))
        samples = discover_samples(root, include_full_face=args.include_full_face)
        from .data import save_manifest

        save_manifest(samples, Path(args.manifest))
        print(json.dumps({"dataset_root": str(root), "samples": len(samples), "manifest": args.manifest}, indent=2))
        return

    if args.command == "train":
        metrics = train(
            TrainConfig(
                source=args.source,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                include_full_face=args.include_full_face,
                image_size=args.image_size,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                backbone=args.backbone,
                pretrained=args.pretrained,
                patience=args.patience,
                num_workers=args.num_workers,
                device=args.device,
                max_samples_per_identity=args.max_samples_per_identity,
            )
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "finetune":
        metrics = finetune(
            TrainConfig(
                source=args.source,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                include_full_face=args.include_full_face,
                image_size=args.image_size,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                backbone=args.backbone,
                pretrained=False,
                patience=args.patience,
                num_workers=args.num_workers,
                device=args.device,
                init_checkpoint=args.init_checkpoint,
                max_samples_per_identity=args.max_samples_per_identity,
            )
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "train-mc":
        metrics = train_monte_carlo(
            TrainConfig(
                source=args.source,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                include_full_face=args.include_full_face,
                image_size=args.image_size,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                backbone=args.backbone,
                pretrained=args.pretrained,
                patience=args.patience,
                num_workers=args.num_workers,
                device=args.device,
                max_samples_per_identity=args.max_samples_per_identity,
            ),
            runs=args.mc_runs,
            seed_step=args.mc_seed_step,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "evaluate":
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        root = prepare_dataset(Path(args.source), Path(args.cache_dir))
        checkpoint_config = checkpoint["config"]
        samples = discover_samples(
            root,
            include_full_face=checkpoint_config.get("include_full_face", False),
        )
        samples = limit_samples_per_identity(
            samples,
            checkpoint_config.get("max_samples_per_identity"),
            checkpoint_config["seed"],
        )

        from torch.utils.data import DataLoader

        from .data import EarDataset
        from .metrics import identification_metrics

        splits = split_samples(
            samples,
            checkpoint_config["val_ratio"],
            checkpoint_config["test_ratio"],
            checkpoint_config["seed"],
        )
        val_ds = EarDataset(splits["val"], transform=_build_eval_transforms(args.image_size))
        test_ds = EarDataset(splits["test"], transform=_build_eval_transforms(args.image_size))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        device = torch.device(args.device)
        model = build_model(checkpoint["config"]["backbone"], num_classes=len(checkpoint["label_to_index"]), pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        def collect_logits(loader):
            batches = []
            targets = []
            paths = []
            with torch.no_grad():
                for images, batch_targets, batch_paths in loader:
                    batches.append(model(images.to(device)).cpu())
                    targets.extend(batch_targets.tolist())
                    paths.extend(batch_paths)
            return torch.cat(batches), torch.tensor(targets), paths

        val_logits, val_targets, _ = collect_logits(val_loader)
        test_logits, test_targets, all_paths = collect_logits(test_loader)

        # Fit one scalar temperature on validation data only. This changes
        # confidence calibration, never class ranking or identity accuracy.
        log_temperature = torch.nn.Parameter(torch.zeros(1))
        temperature_optimizer = torch.optim.LBFGS(
            [log_temperature], lr=0.05, max_iter=50
        )

        def calibration_closure():
            temperature_optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                val_logits / log_temperature.exp(), val_targets
            )
            loss.backward()
            return loss

        temperature_optimizer.step(calibration_closure)
        temperature = float(log_temperature.exp().detach().item())
        uncalibrated_probabilities = torch.softmax(test_logits, dim=1).numpy()
        probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
        all_targets = test_targets.tolist()
        metrics = identification_metrics(
            probabilities, all_targets, num_bins=args.calibration_bins
        )
        uncalibrated = identification_metrics(
            uncalibrated_probabilities, all_targets, num_bins=args.calibration_bins
        )
        metrics["temperature_scaling"] = {
            "temperature": temperature,
            "fit_split": "validation",
            "uncalibrated_mean_confidence": uncalibrated["mean_confidence"],
            "uncalibrated_expected_calibration_error": uncalibrated[
                "expected_calibration_error"
            ],
            "uncalibrated_negative_log_likelihood": uncalibrated[
                "negative_log_likelihood"
            ],
        }
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        sample_by_path = {str(sample.path): sample for sample in samples}
        index_to_label = {index: label for label, index in checkpoint["label_to_index"].items()}

        def summarize_groups(groups):
            summaries = {}
            for group in sorted(set(groups)):
                indices = np.asarray([value == group for value in groups])
                summaries[group] = {
                    "count": int(indices.sum()),
                    "accuracy": float(
                        (predictions[indices] == np.asarray(all_targets)[indices]).mean()
                    ),
                    "mean_confidence": float(confidence[indices].mean()),
                }
            return summaries

        dataset_groups = [
            index_to_label[target].split("/", 1)[0] for target in all_targets
        ]
        audit_groups = [sample_by_path[path].race for path in all_paths]
        metrics["subgroups"] = {
            "dataset": summarize_groups(dataset_groups),
            "demographic_metadata": summarize_groups(audit_groups),
        }
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "predict":
        if args.temperature <= 0:
            raise ValueError("--temperature must be greater than zero")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        device = torch.device(args.device)
        model = build_model(checkpoint["config"]["backbone"], num_classes=len(checkpoint["label_to_index"]), pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        inv_labels = {idx: label for label, idx in checkpoint["label_to_index"].items()}
        transform = _build_eval_transforms(args.image_size)
        tta_transform = _build_tta_transforms(args.image_size)
        with Image.open(args.image) as image:
            base_image = image.convert("RGB")
        with torch.no_grad():
            if args.tta > 1:
                logits = []
                for pass_index in range(args.tta):
                    current = transform(base_image).unsqueeze(0).to(device) if pass_index == 0 else tta_transform(base_image).unsqueeze(0).to(device)
                    logits.append(model(current))
                logits = torch.stack(logits, dim=0).mean(dim=0)
            else:
                image = transform(base_image).unsqueeze(0).to(device)
                logits = model(image)
            probs = torch.softmax(logits / args.temperature, dim=1).squeeze(0)
            topk = torch.topk(probs, k=min(5, probs.numel()))
        top_probability = float(topk.values[0])
        accepted = args.reject_below is None or top_probability >= args.reject_below
        print(
            json.dumps(
                {
                    "decision": inv_labels[int(topk.indices[0])] if accepted else "no_decision",
                    "accepted": accepted,
                    "rejection_threshold": args.reject_below,
                    "temperature": args.temperature,
                    "top_predictions": [
                        {"label": inv_labels[int(idx)], "probability": float(prob)}
                        for prob, idx in zip(topk.values.tolist(), topk.indices.tolist(), strict=False)
                    ]
                },
                indent=2,
            )
        )
        return

    if args.command == "open-set-test":
        if not 0 <= args.target_far <= 1:
            raise ValueError("--target-far must be between zero and one")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        checkpoint_config = checkpoint["config"]
        known_root = prepare_dataset(Path(args.known_source), Path(args.cache_dir))
        unknown_val_root = prepare_dataset(
            Path(args.unknown_validation_source), Path(args.cache_dir)
        )
        unknown_test_root = prepare_dataset(
            Path(args.unknown_test_source), Path(args.cache_dir)
        )
        known_samples = discover_samples(
            known_root,
            include_full_face=checkpoint_config.get("include_full_face", False),
        )
        known_samples = limit_samples_per_identity(
            known_samples,
            checkpoint_config.get("max_samples_per_identity"),
            checkpoint_config["seed"],
        )
        unknown_val_samples = discover_samples(unknown_val_root)
        unknown_test_samples = discover_samples(unknown_test_root)
        known_splits = split_samples(
            known_samples,
            checkpoint_config["val_ratio"],
            checkpoint_config["test_ratio"],
            checkpoint_config["seed"],
        )
        checkpoint_labels = set(checkpoint["label_to_index"])
        known_labels = {sample.label for sample in known_samples}
        if checkpoint_labels != known_labels:
            raise ValueError(
                "Known-source identities do not match the checkpoint classifier"
            )
        if known_labels & {
            sample.label for sample in unknown_val_samples + unknown_test_samples
        }:
            raise ValueError("Unknown identities overlap the enrolled identities")

        device = torch.device(args.device)
        backbone = checkpoint_config["backbone"]
        model = build_model(
            backbone,
            num_classes=len(checkpoint["label_to_index"]),
            pretrained=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        def encode(samples):
            loader = _embedding_loader(
                samples,
                args.image_size,
                batch_size=args.batch_size,
                num_workers=0,
            )
            embeddings = []
            labels = []
            with torch.no_grad():
                for images, batch_labels, _, _ in loader:
                    features = extract_embeddings(
                        model, backbone, images.to(device)
                    ).cpu()
                    embeddings.append(torch.nn.functional.normalize(features, dim=1))
                    labels.extend(batch_labels)
            return torch.cat(embeddings), labels

        gallery_embeddings, gallery_labels = encode(known_splits["train"])
        identity_order = sorted(set(gallery_labels))
        centroids = []
        for identity in identity_order:
            identity_indices = [
                index
                for index, label in enumerate(gallery_labels)
                if label == identity
            ]
            centroid = gallery_embeddings[identity_indices].mean(dim=0)
            centroids.append(torch.nn.functional.normalize(centroid, dim=0))
        centroid_matrix = torch.stack(centroids)

        def score(samples):
            embeddings, labels = encode(samples)
            similarities = embeddings @ centroid_matrix.T
            best_scores, best_indices = similarities.max(dim=1)
            predictions = [identity_order[index] for index in best_indices.tolist()]
            return (
                np.asarray(best_scores.tolist(), dtype=np.float64),
                np.asarray(predictions),
                np.asarray(labels),
            )

        known_val_scores, known_val_predictions, known_val_labels = score(
            known_splits["val"]
        )
        unknown_val_scores, _, _ = score(unknown_val_samples)
        threshold = _select_open_set_threshold(
            known_val_scores,
            known_val_predictions,
            known_val_labels,
            unknown_val_scores,
            args.target_far,
        )

        known_test_scores, known_test_predictions, known_test_labels = score(
            known_splits["test"]
        )
        unknown_test_scores, _, unknown_test_labels = score(unknown_test_samples)

        def operating_metrics(scores, predictions, labels):
            accepted = scores >= threshold
            correct = predictions == labels
            return {
                "samples": int(labels.size),
                "rank_1_accuracy": float(correct.mean()),
                "acceptance_rate": float(accepted.mean()),
                "identification_rate": float((accepted & correct).mean()),
                "false_identification_rate": float((accepted & ~correct).mean()),
                "rejection_rate": float((~accepted).mean()),
            }

        results = {
            "threshold_selection": {
                "target_false_accept_rate": args.target_far,
                "cosine_similarity_threshold": float(threshold),
                "unknown_validation_identities": len(
                    {sample.label for sample in unknown_val_samples}
                ),
                "unknown_validation_samples": len(unknown_val_samples),
                "observed_validation_false_accept_rate": float(
                    (unknown_val_scores >= threshold).mean()
                ),
                "known_validation_identification_rate": float(
                    (
                        (known_val_predictions == known_val_labels)
                        & (known_val_scores >= threshold)
                    ).mean()
                ),
            },
            "known_test": operating_metrics(
                known_test_scores, known_test_predictions, known_test_labels
            ),
            "unknown_test": {
                "identities": len(set(unknown_test_labels.tolist())),
                "samples": int(unknown_test_labels.size),
                "false_accept_rate": float(
                    (unknown_test_scores >= threshold).mean()
                ),
                "true_reject_rate": float(
                    (unknown_test_scores < threshold).mean()
                ),
                "mean_max_similarity": float(unknown_test_scores.mean()),
                "maximum_similarity": float(unknown_test_scores.max()),
            },
            "gallery": {
                "identities": len(identity_order),
                "images": len(gallery_labels),
            },
        }
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return

    if args.command == "validate":
        root = prepare_dataset(Path(args.source), Path(args.cache_dir))
        samples = discover_samples(root, include_full_face=args.include_full_face)
        from .data import EarDataset
        from torch.utils.data import DataLoader

        splits = split_samples(samples, 0.2, 0.2, 42)
        ds = EarDataset(splits["train"][:4], transform=_build_eval_transforms(args.image_size))
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        device = torch.device(args.device)
        model = build_model(args.backbone, num_classes=len({s.label for s in samples}), pretrained=args.pretrained).to(device)
        images, targets, _ = next(iter(loader))
        with torch.no_grad():
            outputs = model(images.to(device))
        print(json.dumps({"samples": len(samples), "batch_shape": list(images.shape), "logits_shape": list(outputs.shape)}, indent=2))
        return

    if args.command == "cross-test":
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        root = prepare_dataset(Path(args.source), Path(args.cache_dir))
        samples = discover_samples(root, include_full_face=args.include_full_face)
        if not samples:
            raise ValueError(f"No images found in {root}")

        device = torch.device(args.device)
        backbone = checkpoint["config"]["backbone"]
        model = build_model(backbone, num_classes=len(checkpoint["label_to_index"]), pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        by_subject: dict[str, list] = {}
        for sample in sorted(samples, key=lambda s: str(s.path)):
            by_subject.setdefault(sample.subject_id, []).append(sample)

        gallery_samples = []
        probe_samples = []
        for subject_id, subject_samples in by_subject.items():
            if len(subject_samples) < 2:
                continue
            gallery_samples.append(subject_samples[0])
            probe_samples.extend(subject_samples[1:])

        if not gallery_samples or not probe_samples:
            raise ValueError("Need at least two images per subject for cross-dataset evaluation")

        gallery_loader = _embedding_loader(gallery_samples, args.image_size, batch_size=16, num_workers=0)
        probe_loader = _embedding_loader(probe_samples, args.image_size, batch_size=16, num_workers=0)

        def encode(loader):
            embeddings = []
            labels = []
            paths = []
            with torch.no_grad():
                for images, subject_ids, _, batch_paths in loader:
                    images = images.to(device)
                    feats = extract_embeddings(model, backbone, images)
                    feats = _mean_normalize(feats.cpu())
                    embeddings.append(feats)
                    labels.extend(list(subject_ids))
                    paths.extend(list(batch_paths))
            return torch.cat(embeddings, dim=0), labels, paths

        gallery_emb, gallery_ids, _ = encode(gallery_loader)
        probe_emb, probe_ids, probe_paths = encode(probe_loader)

        gallery_emb = torch.nn.functional.normalize(gallery_emb, dim=1)
        probe_emb = torch.nn.functional.normalize(probe_emb, dim=1)
        sims = probe_emb @ gallery_emb.T
        nn_indices = sims.argmax(dim=1).tolist()
        preds = [gallery_ids[idx] for idx in nn_indices]
        correct = sum(int(p == t) for p, t in zip(preds, probe_ids, strict=False))
        accuracy = correct / len(probe_ids)
        top_similarities = sims.max(dim=1).values.tolist()
        print(
            json.dumps(
                {
                    "gallery_subjects": len(gallery_ids),
                    "probe_samples": len(probe_ids),
                    "cross_dataset_top1_accuracy": accuracy,
                    "mean_top_similarity": float(np.mean(top_similarities)),
                    "sample_probe_paths": probe_paths[:5],
                },
                indent=2,
            )
        )
        return

    raise RuntimeError(f"Unknown command: {args.command}")
