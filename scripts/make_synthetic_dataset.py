from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.synthetic import create_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic lens/non-lens dataset.")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-per-class", type=int, default=48)
    parser.add_argument("--val-per-class", type=int, default=16)
    parser.add_argument("--test-per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = create_synthetic_dataset(
        output_dir=args.output_dir,
        image_size=args.image_size,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )
    print(f"Synthetic dataset written to {manifest_path}")


if __name__ == "__main__":
    main()

