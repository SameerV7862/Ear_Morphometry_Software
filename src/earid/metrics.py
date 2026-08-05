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
