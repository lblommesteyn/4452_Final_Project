from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from lensing.metrics import expected_calibration_error, stable_sigmoid

def _load_probs(data: dict[str, Any], calibrated: bool = False) -> np.ndarray:
    logits = np.asarray(data["logits"], dtype=np.float32)
    if calibrated:
        temperature = data.get("temperature")
        if temperature is not None and float(temperature) > 0:
            logits = logits / float(temperature)
    return stable_sigmoid(logits)


def _bin_stats(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Return (mean_accuracy, mean_confidence, count) per bin.
    accs, confs, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi)
        counts.append(int(mask.sum()))
        if mask.sum() == 0:
            accs.append(np.nan)
            confs.append(np.nan)
        else:
            accs.append(float(targets[mask].mean()))
            confs.append(float(probabilities[mask].mean()))
    return np.array(accs), np.array(confs), np.array(counts)


def plot_pr_curve(
    results: dict[str, dict[str, Any]],
    output_path: Path,
    title: str = "Precision–Recall Curve",
    use_calibrated: bool = False,
) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))

    last_targets: np.ndarray | None = None
    for model_name, data in results.items():
        probs = _load_probs(data, calibrated=use_calibrated)
        targets = np.asarray(data["targets"], dtype=np.int32)
        last_targets = targets

        precision, recall, _ = precision_recall_curve(targets, probs)
        ap = average_precision_score(targets, probs)
        ax.plot(recall, precision, linewidth=1.8, label=f"{model_name} (AP = {ap:.4f})")

    # Random-classifier baseline at dataset prevalence
    if last_targets is not None and len(last_targets) > 0:
        prevalence = float(last_targets.mean())
        ax.axhline(
            prevalence,
            color="grey",
            linestyle="--",
            linewidth=0.9,
            label=f"Random classifier (prev = {prevalence:.2f})",
        )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_reliability_diagram(
    results: dict[str, dict[str, Any]],
    output_path: Path,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> Path:
    output_path = Path(output_path)
    n_models = len(results)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5.5 * n_models, 5.0),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]  # shape (n_models,)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]

    for ax, (model_name, data) in zip(axes, results.items()):
        targets = np.asarray(data["targets"], dtype=np.int32)
        probs_raw = _load_probs(data, calibrated=False)
        ece_raw = expected_calibration_error(probs_raw, targets, n_bins=n_bins)

        accs_raw, confs_raw, counts_raw = _bin_stats(probs_raw, targets, bin_edges)

        # Secondary axis for histogram counts
        ax2 = ax.twinx()
        ax2.bar(
            bin_centers,
            counts_raw,
            width=bin_width * 0.85,
            alpha=0.12,
            color="steelblue",
            label="Count (raw)",
        )
        ax2.set_ylabel("Samples per bin", fontsize=8, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="Perfect calibration")

        # Raw calibration curve
        valid = ~np.isnan(accs_raw)
        ax.plot(
            confs_raw[valid],
            accs_raw[valid],
            "o-",
            linewidth=1.6,
            markersize=5,
            color="steelblue",
            label=f"Raw  (ECE = {ece_raw:.4f})",
        )

        # Calibrated curve (if temperature is available)
        temperature = data.get("temperature")
        if temperature is not None and float(temperature) > 0:
            probs_cal = _load_probs(data, calibrated=True)
            ece_cal = expected_calibration_error(probs_cal, targets, n_bins=n_bins)
            accs_cal, confs_cal, _ = _bin_stats(probs_cal, targets, bin_edges)
            valid_cal = ~np.isnan(accs_cal)
            ax.plot(
                confs_cal[valid_cal],
                accs_cal[valid_cal],
                "s--",
                linewidth=1.6,
                markersize=5,
                color="darkorange",
                label=f"Calibrated  (ECE = {ece_cal:.4f})",
            )

        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("Fraction of positives", fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def format_low_fpr_table(results: dict[str, dict[str, Any]]) -> str:
    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "n/a"

    header = (
        "| Model | P@FPR=1% | P@FPR=0.1% | Recall@Top-10 | Recall@Top-25 |\n"
        "|---|---:|---:|---:|---:|"
    )
    rows = [header]
    for model_name, data in results.items():
        m = data.get("metrics", {})
        rows.append(
            f"| {model_name} "
            f"| {_fmt(m.get('precision_at_fpr_1e-2'))} "
            f"| {_fmt(m.get('precision_at_fpr_1e-3'))} "
            f"| {_fmt(m.get('recall_at_top_10'))} "
            f"| {_fmt(m.get('recall_at_top_25'))} |"
        )
    return "\n".join(rows)
