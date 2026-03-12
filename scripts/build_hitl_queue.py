from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.active_learning import build_hitl_queue


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human review queue from prediction scores.")
    parser.add_argument("--predictions", required=True, help="CSV created by the training pipeline.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--comparison-predictions", default=None, help="Optional second CSV for disagreement ranking.")
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = str(Path(args.predictions).with_name("hitl_queue.csv"))

    path = build_hitl_queue(
        predictions_path=args.predictions,
        output_path=output,
        top_k=args.top_k,
        comparison_predictions_path=args.comparison_predictions,
    )
    print(f"HITL queue written to {path}")


if __name__ == "__main__":
    main()

