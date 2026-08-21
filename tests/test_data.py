from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image

from src.earid.data import (
    discover_samples,
    limit_samples_per_identity,
    split_samples,
)


class IdentityParsingTests(TestCase):
    def test_ami_filenames_become_individual_subjects(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / "ami" / "subset-1"
            subset.mkdir(parents=True)
            for name in ("001_front_ear.jpg", "001_up_ear.jpg", "002_front_ear.jpg"):
                Image.new("RGB", (8, 8)).save(subset / name)

            samples = discover_samples(root)

            self.assertEqual({"ami/001", "ami/002"}, {sample.label for sample in samples})

    def test_two_image_identity_is_not_forced_into_evaluation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "subject"
            subject.mkdir()
            for name in ("subject_left.jpg", "subject_right.jpg"):
                Image.new("RGB", (8, 8)).save(subject / name)

            splits = split_samples(discover_samples(root), 0.2, 0.2, 42)

            self.assertEqual(2, len(splits["train"]))
            self.assertEqual([], splits["val"])
            self.assertEqual([], splits["test"])

    def test_babyear4k_zip_layout_is_detected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (root / "health_data.csv").write_text("id\n", encoding="utf-8")
            for name in ("0001_L.jpg", "0001_R.jpg"):
                Image.new("RGB", (8, 8)).save(images / name)

            samples = discover_samples(root)

            self.assertEqual({"babyear4k/0001"}, {sample.label for sample in samples})
            self.assertEqual({"BabyEar4k"}, {sample.race for sample in samples})

    def test_earvn_person_folder_becomes_one_identity(self):
        with TemporaryDirectory(prefix="earvn-") as directory:
            root = Path(directory)
            subject = root / "EarVN1.0 dataset" / "Images" / "001.ALI_HD"
            subject.mkdir(parents=True)
            for name in ("001 (1).jpg", "001 (2).jpg"):
                Image.new("RGB", (8, 8)).save(subject / name)

            samples = discover_samples(root)

            self.assertEqual(
                {"earvn/001.ALI_HD"}, {sample.label for sample in samples}
            )
            self.assertEqual({"EarVN1.0"}, {sample.race for sample in samples})

    def test_identity_cap_is_deterministic(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "subject"
            subject.mkdir()
            for index in range(10):
                Image.new("RGB", (8, 8)).save(subject / f"subject_{index}.jpg")
            samples = discover_samples(root)

            first = limit_samples_per_identity(samples, 3, 42)
            second = limit_samples_per_identity(samples, 3, 42)

            self.assertEqual(3, len(first))
            self.assertEqual(
                [sample.path for sample in first],
                [sample.path for sample in second],
            )

    def test_ibug_collection_folder_becomes_one_identity(self):
        with TemporaryDirectory(prefix="ibug-") as directory:
            root = Path(directory)
            subject = root / "CollectionB" / "Fiona_Gubelmann"
            subject.mkdir(parents=True)
            for name in ("0001.png", "0002.png"):
                Image.new("RGB", (8, 8)).save(subject / name)

            samples = discover_samples(root)

            self.assertEqual(
                {"ibug/Fiona_Gubelmann"}, {sample.label for sample in samples}
            )
            self.assertEqual({"iBUG"}, {sample.race for sample in samples})
