from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.active_learning import build_hitl_queue
from lensing.config import load_config
from lensing.training import evaluate_checkpoint, run_training
from lensing.utils import save_json


def parse_binary_label(raw: str) -> int:
    value = raw.strip().lower()
    positive = {"1", "true", "yes", "y", "lens", "positive", "pos"}
    negative = {"0", "false", "no", "n", "non_lens", "non-lens", "negative", "neg"}
    if value in positive:
        return 1
    if value in negative:
        return 0
    raise ValueError(
        f"Unsupported label value '{raw}'. Use binary values such as 0/1 or lens/non_lens."
    )


def _normalized_path(path: Path) -> str:
    return str(path.resolve()).casefold()


def _load_review_labels(review_csv: Path) -> dict[str, int]:
    with review_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Review CSV has no header row: {review_csv}")

        path_column = next(
            (name for name in ("path", "image_path") if name in reader.fieldnames), None
        )
        label_column = next(
            (
                name
                for name in ("reviewed_label", "new_label", "label")
                if name in reader.fieldnames
            ),
            None,
        )
        if path_column is None or label_column is None:
            raise ValueError(
                "Review CSV must include path/image_path and reviewed_label/new_label/label columns."
            )

        labels: dict[str, int] = {}
        for row in reader:
            raw_path = (row.get(path_column) or "").strip()
            raw_label = (row.get(label_column) or "").strip()
            if not raw_path or not raw_label:
                continue
            labels[_normalized_path(Path(raw_path))] = parse_binary_label(raw_label)

    if not labels:
        raise ValueError(f"No usable reviewed labels found in {review_csv}")
    return labels


def _simulate_review_csv_from_queue(
    queue_path: Path,
    output_review_csv: Path,
    flip_fraction: float,
    seed: int,
) -> dict[str, int]:
    if flip_fraction < 0.0 or flip_fraction > 1.0:
        raise ValueError("--simulate-flip-fraction must be in [0, 1].")

    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise ValueError(f"Queue CSV is empty: {queue_path}")
    if fieldnames is None:
        raise ValueError(f"Queue CSV has no header row: {queue_path}")
    if "label" not in fieldnames:
        raise ValueError(f"Queue CSV must include a 'label' column: {queue_path}")

    reviewed_column = "reviewed_label"
    output_fieldnames = list(fieldnames)
    if reviewed_column not in output_fieldnames:
        output_fieldnames.append(reviewed_column)

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    flip_count = int(round(len(rows) * flip_fraction))
    flip_indices = set(rng.sample(indices, k=flip_count)) if flip_count > 0 else set()

    changed = 0
    for index, row in enumerate(rows):
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(
                f"Queue label must be binary 0/1, got {label} at row {index + 2}."
            )
        reviewed_label = 1 - label if index in flip_indices else label
        if reviewed_label != label:
            changed += 1
        row[reviewed_column] = str(reviewed_label)

    output_review_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "num_queue_rows": len(rows),
        "labels_flipped": changed,
    }


def apply_reviewed_labels_to_manifest(
    manifest_path: Path,
    review_labels: dict[str, int],
    output_manifest_path: Path,
    train_split_only: bool = True,
) -> dict[str, int]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise ValueError(f"Manifest is missing a header row: {manifest_path}")
    required = {"image_path", "label", "split"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    manifest_dir = manifest_path.parent
    matched = 0
    changed = 0
    skipped_non_train = 0

    for row in rows:
        image_path = Path(row["image_path"])
        resolved_image_path = (
            image_path if image_path.is_absolute() else (manifest_dir / image_path)
        )
        key = _normalized_path(resolved_image_path)

        if key not in review_labels:
            continue
        if train_split_only and row.get("split") != "train":
            skipped_non_train += 1
            continue

        matched += 1
        new_label = review_labels[key]
        current_label = int(row["label"])
        if current_label != new_label:
            row["label"] = str(new_label)
            changed += 1

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "reviewed_paths": len(review_labels),
        "matched_manifest_rows": matched,
        "labels_changed": changed,
        "skipped_non_train_matches": skipped_non_train,
    }


def _metric_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name in sorted(set(before.keys()) & set(after.keys())):
        before_value = before.get(name)
        after_value = after.get(name)
        if isinstance(before_value, (int, float)) and isinstance(
            after_value, (int, float)
        ):
            deltas[name] = float(after_value) - float(before_value)
    return deltas


def infer_predictions_path(predictions_arg: str | None, checkpoint_path: str) -> Path:
    if predictions_arg:
        return Path(predictions_arg)
    default_path = Path(checkpoint_path).with_name("test_predictions.csv")
    if not default_path.exists():
        raise FileNotFoundError(
            "Could not infer predictions CSV from checkpoint directory. "
            "Pass --predictions explicitly."
        )
    return default_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one HITL round: build queue -> apply reviewed labels -> retrain -> report before/after metrics."
        )
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML experiment config."
    )
    parser.add_argument(
        "--review-csv", default=None, help="CSV with human-reviewed labels."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional baseline checkpoint. If omitted, baseline is trained.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Baseline predictions CSV used to build the queue.",
    )
    parser.add_argument(
        "--comparison-predictions",
        default=None,
        help="Optional second predictions CSV for disagreement ranking.",
    )
    parser.add_argument(
        "--queue-output", default=None, help="Output path for generated HITL queue CSV."
    )
    parser.add_argument("--top-k", type=int, default=50, help="Queue size.")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split for before/after comparison.",
    )
    parser.add_argument(
        "--updated-manifest",
        default=None,
        help="Path to write relabeled manifest. Defaults to <manifest_stem>_hitl_round1.csv.",
    )
    parser.add_argument(
        "--retrain-output-dir",
        default=None,
        help="Output directory for retrained model. Defaults to <config_output_dir>_hitl_round1.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="JSON report path. Defaults to <retrain_output_dir>/hitl_round_report.json.",
    )
    parser.add_argument(
        "--allow-non-train-updates",
        action="store_true",
        help="Allow reviewed labels to modify val/test rows as well.",
    )
    parser.add_argument(
        "--simulate-hitl-data",
        action="store_true",
        help="Create a simulated review CSV from the queue and auto-fill reviewed_label.",
    )
    parser.add_argument(
        "--simulate-flip-fraction",
        type=float,
        default=0.2,
        help="Fraction of queue labels to flip when --simulate-hitl-data is enabled.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.checkpoint is None:
        baseline_train_summary = run_training(base_config)
        baseline_checkpoint = str(baseline_train_summary["checkpoint_path"])
    else:
        baseline_checkpoint = args.checkpoint

    predictions_path = infer_predictions_path(args.predictions, baseline_checkpoint)
    queue_output = (
        Path(args.queue_output)
        if args.queue_output
        else Path(base_config.output_dir) / f"hitl_queue_top{args.top_k}.csv"
    )
    queue_path = build_hitl_queue(
        predictions_path=predictions_path,
        output_path=queue_output,
        top_k=args.top_k,
        comparison_predictions_path=args.comparison_predictions,
    )

    simulated_review_summary = None
    if args.simulate_hitl_data:
        review_csv_path = (
            Path(args.review_csv)
            if args.review_csv
            else queue_path.with_name(f"{queue_path.stem}_simulated_review.csv")
        )
        simulated_review_summary = _simulate_review_csv_from_queue(
            queue_path=queue_path,
            output_review_csv=review_csv_path,
            flip_fraction=args.simulate_flip_fraction,
            seed=42,
        )
    else:
        if not args.review_csv:
            raise ValueError(
                "--review-csv is required unless --simulate-hitl-data is provided."
            )
        review_csv_path = Path(args.review_csv)

    review_labels = _load_review_labels(review_csv_path)
    manifest_path = Path(base_config.data.manifest_path)
    updated_manifest_path = (
        Path(args.updated_manifest)
        if args.updated_manifest
        else manifest_path.with_name(
            f"{manifest_path.stem}_hitl_round1{manifest_path.suffix}"
        )
    )
    manifest_update_summary = apply_reviewed_labels_to_manifest(
        manifest_path=manifest_path,
        review_labels=review_labels,
        output_manifest_path=updated_manifest_path,
        train_split_only=not args.allow_non_train_updates,
    )

    before_eval = evaluate_checkpoint(
        base_config, baseline_checkpoint, split=args.split
    )

    retrain_config = deepcopy(base_config)
    retrain_config.data.manifest_path = str(updated_manifest_path)
    retrain_config.output_dir = (
        args.retrain_output_dir or f"{base_config.output_dir.rstrip('/\\')}_hitl_round1"
    )
    retrain_summary = run_training(retrain_config)
    after_eval = evaluate_checkpoint(
        retrain_config, retrain_summary["checkpoint_path"], split=args.split
    )

    report_path = (
        Path(args.report_output)
        if args.report_output
        else Path(retrain_config.output_dir) / "hitl_round_report.json"
    )
    report = {
        "config": args.config,
        "split": args.split,
        "baseline_checkpoint": baseline_checkpoint,
        "baseline_predictions": str(predictions_path),
        "queue_path": str(queue_path),
        "review_csv": str(review_csv_path),
        "simulate_hitl_data": args.simulate_hitl_data,
        "simulate_review_summary": simulated_review_summary,
        "updated_manifest": str(updated_manifest_path),
        "manifest_update_summary": manifest_update_summary,
        "before": before_eval,
        "after": after_eval,
        "metric_deltas": _metric_deltas(before_eval["metrics"], after_eval["metrics"]),
        "new_checkpoint": retrain_summary["checkpoint_path"],
        "new_output_dir": retrain_config.output_dir,
    }
    save_json(report_path, report)

    print(json.dumps(report, indent=2))
    print(f"HITL round report written to {report_path}")


if __name__ == "__main__":
    main()
