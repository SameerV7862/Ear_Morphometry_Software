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


def infer_identity(root: Path, path: Path) -> tuple[str, str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    stem_subject = path.stem.split("_")[0]
    root_name = root.name.lower()
    top_level = parts[0].lower() if parts else ""
    babyear4k_layout = (root / "health_data.csv").is_file() and (
        root / "images"
    ).is_dir()

    if top_level == "ami" or top_level.startswith("subset-") or "ami" in root_name:
        return f"ami/{stem_subject}", "AMI", stem_subject

    if top_level == "earvn" or "earvn" in root_name:
        subject_id = path.parent.name
        return f"earvn/{subject_id}", "EarVN1.0", subject_id

    if top_level in {"ibug", "collectionb"} or "ibug" in root_name:
        subject_id = path.parent.name
        return f"ibug/{subject_id}", "iBUG", subject_id

    if babyear4k_layout or top_level == "babyear4k" or "babyear4k" in root_name:
        return f"babyear4k/{stem_subject}", "BabyEar4k", stem_subject

    if top_level == "current" and len(parts) >= 4:
        subject_id = parts[-2]
        return f"current/{subject_id}", parts[-3], subject_id

    if len(parts) >= 3:
        label = "/".join(parts[:-1])
        return label, parts[-3], parts[-2]
    if len(parts) == 2:
        subject_id = parts[0] if parts[0].isdigit() else stem_subject
        return subject_id, parts[0], subject_id
    if len(parts) == 1:
        return stem_subject, root.name, stem_subject
    raise ValueError(f"Cannot infer identity from {path}")


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
    discovered: list[tuple[Path, str, str, str, str]] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            path = Path(dirpath) / filename
            if not is_image_file(path):
                continue
            view_type = infer_view_type(path)
            if not include_full_face and view_type == "full_face":
                continue
            label, race, subject_id = infer_identity(root, path)
            discovered.append((path, label, race, subject_id, view_type))

    labels = sorted({item[1] for item in discovered})
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    samples: list[Sample] = []
    for path, label, race, subject_id, view_type in sorted(discovered):
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


def limit_samples_per_identity(
    samples: list[Sample], maximum: int | None, seed: int
) -> list[Sample]:
    if maximum is None:
        return samples
    if maximum < 1:
        raise ValueError("maximum samples per identity must be at least one")

    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    selected = []
    for label, label_samples in sorted(by_label.items()):
        ordered = sorted(label_samples, key=lambda sample: str(sample.path))
        if len(ordered) > maximum:
            rng = random.Random(f"{seed}:{label}")
            ordered = sorted(
                rng.sample(ordered, maximum), key=lambda sample: str(sample.path)
            )
        selected.extend(ordered)
    return selected


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
        # Classes with fewer than 3 images cannot support a valid held-out
        # evaluation (e.g. a subject with only one left-ear and one right-ear
        # image would be trained on one ear and tested on a physically
        # different ear). Use them as training-only diversity data.
        if n < 3:
            train.extend(shuffled)
            continue
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1
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
