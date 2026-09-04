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
