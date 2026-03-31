from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.config import ExperimentConfig, load_config
from lensing.utils import ensure_dir, save_json


@dataclass
class AblationSpec:
    label: str
    description: str
    config_path: Path
    overrides: dict[str, dict[str, Any]]


def _apply_overrides(config: ExperimentConfig, overrides: dict[str, dict[str, Any]]) -> None:
    for section, values in overrides.items():
        if not values:
            continue
        target: Any
        if section == "experiment":
            target = config
        elif section == "data":
            target = config.data
        elif section == "model":
            target = config.model
        elif section == "training":
            target = config.training
        else:
            raise ValueError(f"Unknown override section '{section}'")
        for key, value in values.items():
            setattr(target, key, value)


def _prepare_config(spec: AblationSpec, output_root: Path) -> ExperimentConfig:
    config = load_config(spec.config_path)
    config.name = spec.label
    config.output_dir = str(output_root / spec.label)
    _apply_overrides(config, spec.overrides)
    return config


def _build_specs(args: argparse.Namespace) -> list[AblationSpec]:
    resnet_cfg = Path(args.resnet_config)
    vit_cfg = Path(args.vit_config)
    return [
        AblationSpec(
            label="resnet_weighted_calibrated",
            description="ResNet-18 baseline with weighted sampler, focal loss, and temperature scaling.",
            config_path=resnet_cfg,
            overrides={
                "training": {"calibration": True, "focal_gamma": 1.5},
                "data": {"use_weighted_sampler": True},
            },
        ),
        AblationSpec(
            label="resnet_weighted_raw",
            description="ResNet-18 baseline with weighted sampler but without calibration (raw probabilities).",
            config_path=resnet_cfg,
            overrides={
                "training": {"calibration": False, "focal_gamma": 1.5},
                "data": {"use_weighted_sampler": True},
            },
        ),
        AblationSpec(
            label="resnet_weighted_bce",
            description="ResNet-18 with weighted sampler but vanilla BCE (gamma=0) to isolate loss effects.",
            config_path=resnet_cfg,
            overrides={
                "training": {"calibration": True, "focal_gamma": 0.0, "positive_class_weight": 1.0},
                "data": {"use_weighted_sampler": True},
            },
        ),
        AblationSpec(
            label="resnet_unweighted_bce",
            description="ResNet-18 with vanilla BCE (gamma=0), no weighted sampler, heavier positive class weight.",
            config_path=resnet_cfg,
            overrides={
                "training": {"calibration": True, "focal_gamma": 0.0, "positive_class_weight": 3.0},
                "data": {"use_weighted_sampler": False},
            },
        ),
        AblationSpec(
            label="resnet_unweighted_focal",
            description="ResNet-18 without weighted sampler but using focal loss plus positive-class weighting.",
            config_path=resnet_cfg,
            overrides={
                "training": {"calibration": True, "focal_gamma": 1.5, "positive_class_weight": 3.0},
                "data": {"use_weighted_sampler": False},
            },
        ),
        AblationSpec(
            label="vit_weighted_calibrated",
            description="ViT-B/16 baseline with weighted sampler and calibration enabled.",
            config_path=vit_cfg,
            overrides={
                "training": {"calibration": True, "focal_gamma": 1.5},
                "data": {"use_weighted_sampler": True},
            },
        ),
    ]


def _run_training(config: ExperimentConfig) -> dict[str, Any]:
    from lensing.training import run_training as _run_training_impl

    return _run_training_impl(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standard ablation set covering ResNet vs. ViT, "
            "raw vs. calibrated probabilities, and an imbalance handling variant."
        )
    )
    parser.add_argument(
        "--resnet-config",
        default="configs/resnet18_default.yaml",
        help="Path to the base ResNet config used for ablations.",
    )
    parser.add_argument(
        "--vit-config",
        default="configs/vit_b16_default.yaml",
        help="Path to the base ViT config used for ablations.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/ablations",
        help="Directory where ablation run outputs and summaries will be stored.",
    )
    args = parser.parse_args()

    output_root = ensure_dir(args.output_root)
    specs = _build_specs(args)

    summaries: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[ablation] Starting {spec.label}: {spec.description}")
        config = _prepare_config(spec, output_root)
        summary = _run_training(config)
        summary["label"] = spec.label
        summary["description"] = spec.description
        summaries.append(summary)

    summary_path = Path(output_root) / "ablations_summary.json"
    save_json(summary_path, {"runs": summaries})
    print(json.dumps({"runs": summaries, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
