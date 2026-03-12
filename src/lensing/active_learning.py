from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return -(clipped * np.log2(clipped) + (1.0 - clipped) * np.log2(1.0 - clipped))


def build_hitl_queue(
    predictions_path: str | Path,
    output_path: str | Path,
    top_k: int = 50,
    comparison_predictions_path: str | Path | None = None,
) -> Path:
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No predictions found in {predictions_path}")

    probabilities = np.array([float(row["probability"]) for row in rows], dtype=np.float32)
    scores = binary_entropy(probabilities)

    if comparison_predictions_path is not None:
        with Path(comparison_predictions_path).open("r", encoding="utf-8", newline="") as handle:
            comparison_rows = list(csv.DictReader(handle))
        if len(comparison_rows) != len(rows):
            raise ValueError("Prediction files must contain the same number of rows for disagreement scoring.")
        comparison_probabilities = np.array(
            [float(row["probability"]) for row in comparison_rows],
            dtype=np.float32,
        )
        scores = np.abs(probabilities - comparison_probabilities)

    order = np.argsort(scores)[::-1][:top_k]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) + ["review_score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in order:
            row = dict(rows[index])
            row["review_score"] = f"{float(scores[index]):.6f}"
            writer.writerow(row)
    return output_path

