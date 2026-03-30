from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.figures import (
    format_low_fpr_table,
    plot_pr_curve,
    plot_reliability_diagram,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest_counts(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[row["split"]][row["source"]] += 1
            counts[row["split"]]["total"] += 1
            counts[row["split"]]["lens"] += int(row["label"]) == 1
            counts[row["split"]]["non_lens"] += int(row["label"]) == 0
    return {split: dict(counter) for split, counter in counts.items()}


def read_top_hitl(path: Path, n: int = 5) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:n]


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    logits = np.array([float(r["logit"]) for r in rows], dtype=np.float32)
    targets = np.array([float(r["label"]) for r in rows], dtype=np.float32)
    return {"logits": logits, "targets": targets}


def format_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def write_report() -> None:
    report_dir = ROOT / "reports"
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "data" / "real_castles_gzh" / "manifest.csv"
    resnet_metrics = load_json(ROOT / "outputs" / "resnet18_real_castles_gzh" / "metrics.json")
    vit_metrics = load_json(ROOT / "outputs" / "vit_b16_real_castles_gzh" / "metrics.json")
    counts = read_manifest_counts(manifest_path)
    resnet_hitl = read_top_hitl(ROOT / "outputs" / "resnet18_real_castles_gzh" / "hitl_queue.csv")
    vit_hitl = read_top_hitl(ROOT / "outputs" / "vit_b16_real_castles_gzh" / "hitl_queue.csv")

    resnet_preds = load_predictions(
        ROOT / "outputs" / "resnet18_real_castles_gzh" / "test_predictions.csv"
    )
    vit_preds = load_predictions(
        ROOT / "outputs" / "vit_b16_real_castles_gzh" / "test_predictions.csv"
    )

    # attach temperature so reliability diagram can draw calibrated curves
    resnet_preds["temperature"] = resnet_metrics.get("temperature")
    vit_preds["temperature"] = vit_metrics.get("temperature")

    # attach scalar metrics for low-FPR table
    resnet_preds["metrics"] = resnet_metrics.get("test_metrics", {})
    vit_preds["metrics"] = vit_metrics.get("test_metrics", {})

    figure_results = {
        "ResNet18": resnet_preds,
        "ViT-B/16": vit_preds,
    }

    pr_curve_path = figures_dir / "pr_curve.png"
    reliability_path = figures_dir / "reliability_diagram.png"

    plot_pr_curve(
        figure_results,
        output_path=pr_curve_path,
        title="Precision–Recall Curve — CASTLES/GZH Test Split",
    )
    print(f"Wrote {pr_curve_path}")

    plot_reliability_diagram(
        figure_results,
        output_path=reliability_path,
        n_bins=10,
        title="Reliability Diagram — CASTLES/GZH Test Split",
    )
    print(f"Wrote {reliability_path}")

    low_fpr_table = format_low_fpr_table(figure_results)

    summary = {
        "dataset_counts": counts,
        "resnet18": resnet_metrics,
        "vit_b16": vit_metrics,
        "resnet_hitl_top5": resnet_hitl,
        "vit_hitl_top5": vit_hitl,
        "notes": [
            "Both models reached perfect discrimination on the held-out split built from CASTLES positives and Galaxy Zoo: Hubble negatives.",
            "These scores are likely inflated by dataset separability and source bias, so they should not be interpreted as survey-ready performance.",
            "Raw calibration favoured ResNet on this dataset; both models became nearly perfectly calibrated after temperature scaling because the test set is almost linearly separable.",
            "The PR curve hugs the top-right corner and the reliability diagram shows extreme pre-calibration overconfidence precisely because of this separability; both figures will be more informative on harder data.",
        ],
        "sources": {
            "castles": "https://www.cfa.harvard.edu/castles/",
            "galaxy_zoo": "https://data.galaxyzoo.org/index.html",
            "galaxy_zoo_hubble_samples": "https://data.galaxyzoo.org/data/gzh/samples/low_redshift_disks.html",
        },
    }

    with (report_dir / "real_data_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    def _split_row(split: str) -> str:
        c = counts.get(split, {})
        return (
            f"| {split} "
            f"| {c.get('total', 0)} "
            f"| {c.get('lens', 0)} "
            f"| {c.get('non_lens', 0)} "
            f"| {c.get('castles_hst', 0)} "
            f"| {c.get('galaxy_zoo_hubble', 0)} |"
        )

    def _metric_row(name: str, m: dict) -> str:
        return (
            f"| {name} "
            f"| {format_float(m.get('accuracy'))} "
            f"| {format_float(m.get('precision'))} "
            f"| {format_float(m.get('recall'))} "
            f"| {format_float(m.get('f1'))} "
            f"| {format_float(m.get('roc_auc'))} "
            f"| {format_float(m.get('average_precision'))} "
            f"| {format_float(m.get('ece'))} "
            f"| {format_float(m.get('brier_score'))} |"
        )

    def _cal_row(name: str, metrics_raw: dict, metrics_cal: dict | None, temperature: float | None) -> str:
        cal = metrics_cal or {}
        return (
            f"| {name} "
            f"| {format_float(temperature)} "
            f"| {format_float(metrics_raw.get('ece'))} "
            f"| {format_float(cal.get('ece'))} "
            f"| {format_float(metrics_raw.get('brier_score'))} "
            f"| {format_float(cal.get('brier_score'))} |"
        )

    resnet_tm = resnet_metrics.get("test_metrics", {})
    vit_tm = vit_metrics.get("test_metrics", {})
    resnet_tm_cal = resnet_metrics.get("test_metrics_calibrated")
    vit_tm_cal = vit_metrics.get("test_metrics_calibrated")

    hitl_resnet_lines = "\n".join(
        f"- `{Path(row['path']).name}` label={row['label']} prob={row['probability']} review_score={row['review_score']}"
        for row in resnet_hitl
    )
    hitl_vit_lines = "\n".join(
        f"- `{Path(row['path']).name}` label={row['label']} prob={row['probability']} review_score={row['review_score']}"
        for row in vit_hitl
    )

    markdown = f"""# Real-Data Analysis

## Dataset

Manifest: `data/real_castles_gzh/manifest.csv`

| Split | Total | Lens | Non-lens | CASTLES HST | Galaxy Zoo: Hubble |
|---|---:|---:|---:|---:|---:|
| train | {counts['train']['total']} | {counts['train']['lens']} | {counts['train']['non_lens']} | {counts['train'].get('castles_hst', 0)} | {counts['train'].get('galaxy_zoo_hubble', 0)} |
| val | {counts['val']['total']} | {counts['val']['lens']} | {counts['val']['non_lens']} | {counts['val'].get('castles_hst', 0)} | {counts['val'].get('galaxy_zoo_hubble', 0)} |
| test | {counts['test']['total']} | {counts['test']['lens']} | {counts['test']['non_lens']} | {counts['test'].get('castles_hst', 0)} | {counts['test'].get('galaxy_zoo_hubble', 0)} |

The held-out test split is intentionally imbalanced at 1:4 lens prevalence.

## Test Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg Precision | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | {format_float(resnet_metrics['test_metrics']['accuracy'])} | {format_float(resnet_metrics['test_metrics']['precision'])} | {format_float(resnet_metrics['test_metrics']['recall'])} | {format_float(resnet_metrics['test_metrics']['f1'])} | {format_float(resnet_metrics['test_metrics']['roc_auc'])} | {format_float(resnet_metrics['test_metrics']['average_precision'])} | {format_float(resnet_metrics['test_metrics']['ece'])} | {format_float(resnet_metrics['test_metrics']['brier_score'])} |
| ViT-B/16 | {format_float(vit_metrics['test_metrics']['accuracy'])} | {format_float(vit_metrics['test_metrics']['precision'])} | {format_float(vit_metrics['test_metrics']['recall'])} | {format_float(vit_metrics['test_metrics']['f1'])} | {format_float(vit_metrics['test_metrics']['roc_auc'])} | {format_float(vit_metrics['test_metrics']['average_precision'])} | {format_float(vit_metrics['test_metrics']['ece'])} | {format_float(vit_metrics['test_metrics']['brier_score'])} |

## Calibration

| Model | Temperature | Raw ECE | Calibrated ECE | Raw Brier | Calibrated Brier |
|---|---:|---:|---:|---:|---:|
| ResNet18 | {format_float(resnet_metrics['temperature'])} | {format_float(resnet_metrics['test_metrics']['ece'])} | {format_float(resnet_metrics['test_metrics_calibrated']['ece'])} | {format_float(resnet_metrics['test_metrics']['brier_score'])} | {format_float(resnet_metrics['test_metrics_calibrated']['brier_score'])} |
| ViT-B/16 | {format_float(vit_metrics['temperature'])} | {format_float(vit_metrics['test_metrics']['ece'])} | {format_float(vit_metrics['test_metrics_calibrated']['ece'])} | {format_float(vit_metrics['test_metrics']['brier_score'])} | {format_float(vit_metrics['test_metrics_calibrated']['brier_score'])} |

## Low-FPR Operating Points

{low_fpr_table}

> **Note:** With a perfectly separable test set all values are 1.0.
> These columns become meaningful once harder negatives are included.

## Interpretation

- Both models achieved perfect discrimination on this held-out split.
- ResNet18 had substantially better raw calibration than ViT-B/16 before temperature scaling.
- The top uncertain examples are still mostly positive CASTLES systems with raw probabilities around 0.54 to 0.66, which indicates some intra-lens variation remains despite the perfect thresholded metrics.
- The results are almost certainly inflated by source bias and dataset simplicity: positives are CASTLES lens systems and negatives are Galaxy Zoo: Hubble disk galaxies, which is useful for a proof of pipeline integration but not a realistic estimate of Rubin/LSST survey performance.
- The PR curve hugs the top-right corner and the reliability diagram shows extreme pre-calibration overconfidence precisely because of this separability; both figures will be more informative on harder data.

## Figures

### Precision-Recall Curve

![PR Curve](figures/pr_curve.png)

### Reliability Diagram

![Reliability Diagram](figures/reliability_diagram.png)

## HITL Queue Samples

### ResNet18 top uncertain examples

"""

    for row in resnet_hitl:
        markdown += f"- `{Path(row['path']).name}` label={row['label']} prob={row['probability']} review_score={row['review_score']}\n"

    markdown += "\n### ViT-B/16 top uncertain examples\n\n"
    for row in vit_hitl:
        markdown += f"- `{Path(row['path']).name}` label={row['label']} prob={row['probability']} review_score={row['review_score']}\n"

    markdown += """

## Source Links

- CASTLES lens database: https://www.cfa.harvard.edu/castles/
- Galaxy Zoo data portal: https://data.galaxyzoo.org/index.html
- Galaxy Zoo: Hubble sample images: https://data.galaxyzoo.org/data/gzh/samples/low_redshift_disks.html
"""

    with (report_dir / "real_data_analysis.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(f"Wrote {report_dir / 'real_data_analysis.md'}")


if __name__ == "__main__":
    write_report()
