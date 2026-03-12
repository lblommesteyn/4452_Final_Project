from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.real_dataset import build_real_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build a real strong-lens dataset manifest.")
    parser.add_argument("--output-dir", default="data/real_castles_gzh")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-negative-ratio", type=int, default=4)
    args = parser.parse_args()

    summary = build_real_dataset(
        output_dir=args.output_dir,
        seed=args.seed,
        test_negative_ratio=args.test_negative_ratio,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
