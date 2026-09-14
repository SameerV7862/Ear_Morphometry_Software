import math

import numpy as np
import pytest
import torch
from PIL import Image

from earid.align import (
    NUM_LANDMARKS,
    _crop_around_landmarks,
    _rotate,
    build_landmark_model,
    parse_pts,
)
from earid.models import ArcFaceModel, build_model, extract_embeddings, load_matching_state_dict


def test_parse_pts(tmp_path):
    lines = ["version: 1", f"n_points: {NUM_LANDMARKS}", "{"]
    lines += [f"{i}.0 {i + 0.5}" for i in range(NUM_LANDMARKS)]
    lines.append("}")
    pts = tmp_path / "sample.pts"
    pts.write_text("\n".join(lines))
    points = parse_pts(pts)
    assert points.shape == (NUM_LANDMARKS, 2)
    assert points[3][0] == pytest.approx(3.0)
    assert points[3][1] == pytest.approx(3.5)


def test_rotate_keeps_marked_pixel_under_landmark():
    image = Image.new("RGB", (100, 80), "black")
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            image.putpixel((70 + dx, 20 + dy), (255, 0, 0))
    points = np.array([[70.0, 20.0]], dtype=np.float32)
    rotated, moved = _rotate(image, points, 30.0)
    x, y = int(round(moved[0][0])), int(round(moved[0][1]))
    assert rotated.getpixel((x, y))[0] > 150


def test_crop_around_landmarks_contains_points():
    image = Image.new("RGB", (200, 200))
    points = np.array([[50.0, 60.0], [120.0, 150.0]], dtype=np.float32)
    cropped, shifted = _crop_around_landmarks(image, points, margin=0.2, jitter=0.0)
    assert shifted.min() >= 0
    assert shifted[:, 0].max() <= cropped.width
    assert shifted[:, 1].max() <= cropped.height


def test_landmark_model_output_shape():
    model = build_landmark_model(pretrained=False)
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, NUM_LANDMARKS * 2)
    assert out.min() >= 0 and out.max() <= 1


def test_arcface_forward_train_and_eval():
    model = build_model("resnet18", num_classes=7, pretrained=False, loss="arcface")
    assert isinstance(model, ArcFaceModel)
    x = torch.randn(4, 3, 64, 64)
    targets = torch.tensor([0, 1, 2, 3])
    train_logits = model(x, targets)
    eval_logits = model(x)
    assert train_logits.shape == (4, 7)
    assert eval_logits.shape == (4, 7)
    # margin lowers the true-class logit relative to inference
    rows = torch.arange(4)
    assert (train_logits[rows, targets] <= eval_logits[rows, targets] + 1e-4).all()
    emb = extract_embeddings(model, "resnet18", x)
    assert emb.shape == (4, 512)


def test_arcface_warm_start_from_plain_checkpoint():
    plain = build_model("resnet18", num_classes=7, pretrained=False)
    arc = build_model("resnet18", num_classes=7, pretrained=False, loss="arcface")
    missing, _ = load_matching_state_dict(arc, plain.state_dict())
    # every backbone tensor should transfer; only the arc weight stays missing
    assert missing == ["weight"]


def test_detector_model_output_shape():
    from earid.align import build_detector_model

    model = build_detector_model(pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 4)
    assert out.min() >= 0 and out.max() <= 1


def test_predict_ear_bbox_scales_to_image():
    from PIL import Image

    from earid.align import build_detector_model, predict_ear_bbox

    model = build_detector_model(pretrained=False).eval()
    image = Image.new("RGB", (640, 480))
    bbox = predict_ear_bbox(model, image, 224, torch.device("cpu"))
    assert bbox.shape == (4,)
    assert 0 <= bbox[0] <= 640 and 0 <= bbox[1] <= 480
    assert 0 <= bbox[2] <= 640 and 0 <= bbox[3] <= 480


def test_detect_and_align_returns_output_size():
    from PIL import Image

    from earid.align import build_detector_model, build_landmark_model, detect_and_align

    detector = build_detector_model(pretrained=False).eval()
    landmarks = build_landmark_model(pretrained=False).eval()
    image = Image.new("RGB", (400, 300), (128, 100, 90))
    out = detect_and_align(detector, 224, landmarks, 128, image, torch.device("cpu"), output_size=224)
    assert out.size == (224, 224)


def test_detector_dataset_mixed_kinds(tmp_path):
    import numpy as np
    from PIL import Image

    from earid.align import DetectorDataset

    samples = []
    for i, kind in enumerate(["context", "ear_only"]):
        img_path = tmp_path / f"s{i}.png"
        Image.new("RGB", (200, 160), (90, 90, 90)).save(img_path)
        n = 4 if kind == "context" else 55
        rng = np.random.default_rng(i)
        pts = rng.uniform([40, 30], [150, 130], size=(n, 2))
        lines = [f"version: 1", f"n_points: {n}", "{"] + [f"{x:.2f} {y:.2f}" for x, y in pts] + ["}"]
        img_path.with_suffix(".pts").write_text("\n".join(lines))
        samples.append((img_path, kind))

    for train in (True, False):
        ds = DetectorDataset(samples, image_size=224, train=train)
        for tensor, target in [ds[0], ds[1]]:
            assert tensor.shape == (3, 224, 224)
            assert target.shape == (4,)
            assert (target >= 0).all() and (target <= 1).all()
            assert target[0] <= target[2] and target[1] <= target[3]
