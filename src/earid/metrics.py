from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int64)
    preds = np.asarray(y_pred, dtype=np.int64)
    accuracy = float((truth == preds).mean()) if truth.size else 0.0

    classes = np.unique(np.concatenate([truth, preds])) if truth.size else np.array([], dtype=np.int64)
    f1_scores = []
    for cls in classes:
        tp = float(np.sum((truth == cls) & (preds == cls)))
        fp = float(np.sum((truth != cls) & (preds == cls)))
        fn = float(np.sum((truth == cls) & (preds != cls)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_scores.append((2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def identification_metrics(probabilities, y_true, num_bins: int = 10) -> dict:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.int64)
    if num_bins <= 0:
        raise ValueError("num_bins must be greater than zero")
    if probs.ndim != 2 or probs.shape[0] != truth.size:
        raise ValueError("probabilities must have shape [samples, classes]")
    if truth.size == 0:
        raise ValueError("identification metrics require at least one sample")
    if truth.min() < 0 or truth.max() >= probs.shape[1]:
        raise ValueError("target index is outside the probability class range")

    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    metrics: dict = classification_metrics(truth, predictions)

    ranked = np.argsort(-probs, axis=1)
    for k in (1, 5, 10):
        effective_k = min(k, probs.shape[1])
        metrics[f"top_{k}_accuracy"] = float(
            np.mean([target in row[:effective_k] for target, row in zip(truth, ranked, strict=False)])
        )

    target_probabilities = np.clip(probs[np.arange(truth.size), truth], 1e-12, 1.0)
    metrics["negative_log_likelihood"] = float(-np.log(target_probabilities).mean())
    metrics["mean_confidence"] = float(confidence.mean())

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    calibration_bins = []
    expected_calibration_error = 0.0
    for index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=False)):
        in_bin = (confidence >= lower) & (
            confidence <= upper if index == num_bins - 1 else confidence < upper
        )
        count = int(in_bin.sum())
        bin_accuracy = float((predictions[in_bin] == truth[in_bin]).mean()) if count else 0.0
        bin_confidence = float(confidence[in_bin].mean()) if count else 0.0
        expected_calibration_error += (count / truth.size) * abs(bin_accuracy - bin_confidence)
        calibration_bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": bin_accuracy,
                "mean_confidence": bin_confidence,
            }
        )
    metrics["expected_calibration_error"] = float(expected_calibration_error)
    metrics["calibration_bins"] = calibration_bins

    confidence_order = np.argsort(-confidence)
    correctness = (predictions[confidence_order] == truth[confidence_order]).astype(np.float64)
    risk_coverage = []
    for coverage in np.linspace(0.1, 1.0, 10):
        retained = max(1, int(np.ceil(coverage * truth.size)))
        retained_accuracy = float(correctness[:retained].mean())
        risk_coverage.append(
            {
                "coverage": float(coverage),
                "accuracy": retained_accuracy,
                "risk": 1.0 - retained_accuracy,
                "minimum_confidence": float(confidence[confidence_order[retained - 1]]),
            }
        )
    metrics["risk_coverage"] = risk_coverage
    return metrics
