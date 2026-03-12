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
    args = parser.parse_args()

    config = load_config(args.config)
    summary = evaluate_checkpoint(config, args.checkpoint, split=args.split)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

