"""Explicit operating point, unadjusted window metrics, and paired drift probes."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def calibrate(scores, labels, quantile: float = .99) -> float:
    normals = np.asarray(scores)[np.asarray(labels) == 0]
    if not len(normals):
        raise ValueError("Validation must contain normal windows for calibration.")
    return float(np.quantile(normals, quantile, method="higher"))


def metrics(labels, scores, threshold) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    prediction = scores > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(labels)) == 2
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "average_precision": float(average_precision_score(labels, scores)) if both else None,
            "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
            "prevalence": float(labels.mean()), "false_alerts_per_1000_windows": float(fp / len(labels) * 1000),
            "normal_false_positive_rate": float(fp / max(tn + fp, 1)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "windows": len(labels), "threshold": float(threshold)}


def bootstrap_ap(labels, scores, seed=42, repetitions=200, block_size=50):
    """Moving-block bootstrap across consecutive windows, descriptive not a guarantee."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    rng = np.random.default_rng(seed)
    values = []
    block_size = min(block_size, len(labels))
    for _ in range(repetitions):
        starts = rng.integers(0, len(labels) - block_size + 1, size=int(np.ceil(len(labels) / block_size)))
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:len(labels)]
        if len(np.unique(labels[idx])) == 2:
            values.append(average_precision_score(labels[idx], scores[idx]))
    return [float(v) for v in np.quantile(values, [.025, .975])] if values else None
