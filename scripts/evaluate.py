from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.config import load_config
from lensing.training import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a selected split.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pt checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--perturbations", nargs="+", default=["clean"], help="List of perturbations to evaluate. Options: clean, noise, blur, contrast, shift.",)
    args = parser.parse_args()

    config = load_config(args.config)

    perturbation_settings = {
        "clean": {},
        "noise": {"std": 0.05},
        "blur": {"kernel_size": 3},
        "contrast": {"factor": 1.2},
        "shift": {"shift_x": 2, "shift_y": 2},
    }


    summaries = []
    for perturbation in args.perturbations:
        perturbation = perturbation.lower().strip()
        if perturbation not in perturbation_settings:
            raise ValueError(
                f"Unknown perturbation '{perturbation}'. "
                f"Valid options: {sorted(perturbation_settings.keys())}"
            )

        summary = evaluate_checkpoint(
            config,
            args.checkpoint,
            split=args.split,
            perturbation=None if perturbation == "clean" else perturbation,
            perturbation_kwargs=perturbation_settings[perturbation],
        )
        summaries.append(summary)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

