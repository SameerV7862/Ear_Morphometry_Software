from __future__ import annotations

import csv
import hashlib
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    label_index: int
    race: str
    subject_id: str
    view_type: str


def infer_view_type(path: Path) -> str:
    name = path.name.lower()
    if "full_face" in name:
        return "full_face"
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    if "ear" in name:
        return "ear"
    return "unknown"


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def prepare_dataset(source: Path, cache_dir: Path) -> Path:
    source = source.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    if source.is_dir():
        return source
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported dataset source: {source}")

    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    extracted_root = cache_dir / f"ear_dataset_{digest}"
    marker = extracted_root / ".extracted"
    if marker.exists():
        return extracted_root

    if extracted_root.exists():
        shutil.rmtree(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(extracted_root)
    marker.write_text("ok\n", encoding="utf-8")
    return extracted_root


def discover_samples(root: Path, include_full_face: bool = False) -> list[Sample]:
    root = root.expanduser().resolve()
    labels = set()
    image_paths: list[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            path = Path(dirpath) / filename
            if not is_image_file(path):
                continue
            image_paths.append(path)
            rel = path.relative_to(root)
            parts = rel.parts
            if len(parts) >= 3:
                labels.add("/".join(parts[:-1]))
            elif len(parts) == 2:
                labels.add(parts[0] if parts[0].isdigit() else path.stem.split("_")[0])
            elif len(parts) == 1:
                labels.add(path.stem.split("_")[0])
    labels = sorted(labels)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    samples: list[Sample] = []
    for path in sorted(image_paths):
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 3:
            label = "/".join(parts[:-1])
        elif len(parts) == 2:
            label = parts[0] if parts[0].isdigit() else path.stem.split("_")[0]
        elif len(parts) == 1:
            label = path.stem.split("_")[0]
        else:
            continue
        if label not in label_to_index:
            continue
        view_type = infer_view_type(path)
        if not include_full_face and view_type == "full_face":
            continue
        if len(parts) >= 3:
            race = parts[0]
            subject_id = parts[1]
        elif len(parts) == 2:
            race = parts[0]
            subject_id = label
        else:
            race = root.name
            subject_id = label
        samples.append(
            Sample(
                path=path,
                label=label,
                label_index=label_to_index[label],
                race=race,
                subject_id=subject_id,
                view_type=view_type,
            )
        )
    return samples


def save_manifest(samples: Iterable[Sample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "label_index", "race", "subject_id", "view_type"])
        for sample in samples:
            writer.writerow(
                [
                    str(sample.path),
                    sample.label,
                    sample.label_index,
                    sample.race,
                    sample.subject_id,
                    sample.view_type,
                ]
            )


def split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must be between 0 and 1")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []

    for label_samples in by_label.values():
        shuffled = label_samples[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1 if n >= 3 else 0
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("Split ratios leave no training samples for at least one class")
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


class EarDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label_index, str(sample.path)


def build_manifest_digest(samples: Iterable[Sample]) -> str:
    hasher = hashlib.sha1()
    for sample in sorted(samples, key=lambda s: str(s.path)):
        hasher.update(str(sample.path).encode("utf-8"))
        hasher.update(sample.label.encode("utf-8"))
        hasher.update(sample.view_type.encode("utf-8"))
    return hasher.hexdigest()
