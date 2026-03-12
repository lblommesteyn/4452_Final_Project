from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    positive_mask = logits >= 0
    negative_mask = ~positive_mask
    probabilities = np.empty_like(logits, dtype=np.float32)
    probabilities[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_logits = np.exp(logits[negative_mask])
    probabilities[negative_mask] = exp_logits / (1.0 + exp_logits)
    return probabilities


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if not np.any(in_bin):
            continue
        bin_accuracy = targets[in_bin].mean()
        bin_confidence = probabilities[in_bin].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * np.mean(in_bin)
    return float(ece)


def precision_at_fixed_fpr(
    probabilities: np.ndarray,
    targets: np.ndarray,
    fixed_fpr: float,
) -> float | None:
    negatives = np.where(targets == 0)[0]
    positives = np.where(targets == 1)[0]
    if len(negatives) == 0 or len(positives) == 0:
        return None

    thresholds = np.unique(probabilities)[::-1]
    best_precision = None
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        false_positives = int(np.sum((predictions == 1) & (targets == 0)))
        true_positives = int(np.sum((predictions == 1) & (targets == 1)))
        current_fpr = false_positives / len(negatives)
        if current_fpr <= fixed_fpr and true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            if best_precision is None or precision > best_precision:
                best_precision = precision
    return None if best_precision is None else float(best_precision)


def recall_at_top_k(probabilities: np.ndarray, targets: np.ndarray, top_k: int) -> float | None:
    if len(probabilities) == 0:
        return None
    positives = int(np.sum(targets == 1))
    if positives == 0:
        return None
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    recovered = int(np.sum(targets[top_indices] == 1))
    return float(recovered / positives)


def safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return None


def summarize_binary_metrics(
    logits: Iterable[float],
    targets: Iterable[float],
    threshold: float = 0.5,
    loss: float | None = None,
) -> dict[str, float | None]:
    logits = np.asarray(list(logits), dtype=np.float32)
    targets = np.asarray(list(targets), dtype=np.int32)
    probabilities = stable_sigmoid(logits)
    predictions = (probabilities >= threshold).astype(np.int32)

    return {
        "loss": loss,
        "num_examples": int(len(targets)),
        "positive_rate": float(targets.mean()) if len(targets) > 0 else None,
        "accuracy": safe_metric(accuracy_score, targets, predictions),
        "precision": safe_metric(precision_score, targets, predictions, zero_division=0),
        "recall": safe_metric(recall_score, targets, predictions, zero_division=0),
        "f1": safe_metric(f1_score, targets, predictions, zero_division=0),
        "roc_auc": safe_metric(roc_auc_score, targets, probabilities),
        "average_precision": safe_metric(average_precision_score, targets, probabilities),
        "brier_score": safe_metric(brier_score_loss, targets, probabilities),
        "ece": expected_calibration_error(probabilities, targets) if len(targets) > 0 else None,
        "precision_at_fpr_1e-2": precision_at_fixed_fpr(probabilities, targets, 1e-2),
        "precision_at_fpr_1e-3": precision_at_fixed_fpr(probabilities, targets, 1e-3),
        "recall_at_top_10": recall_at_top_k(probabilities, targets, 10),
        "recall_at_top_25": recall_at_top_k(probabilities, targets, 25),
    }
