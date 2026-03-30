from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.active_learning import build_hitl_queue
from lensing.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from lensing.figures import (
    format_low_fpr_table,
    plot_pr_curve,
    plot_reliability_diagram,
)
from lensing.synthetic import create_synthetic_dataset
from lensing.training import evaluate_checkpoint, run_training


class SmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = ROOT / "tmp_smoke"
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_end_to_end_training_pipeline(self) -> None:
        manifest_path = create_synthetic_dataset(
            output_dir=self.test_root / "data",
            image_size=96,
            train_per_class=8,
            val_per_class=4,
            test_per_class=4,
            seed=7,
        )
        config = ExperimentConfig(
            name="smoke_resnet18",
            output_dir=str(self.test_root / "outputs"),
            seed=7,
            data=DataConfig(
                manifest_path=str(manifest_path),
                image_size=96,
                batch_size=4,
                num_workers=0,
            ),
            model=ModelConfig(
                name="resnet18",
                image_size=96,
                pretrained=False,
                dropout=0.1,
                freeze_backbone=False,
            ),
            training=TrainingConfig(
                epochs=2,
                learning_rate=1e-3,
                weight_decay=1e-4,
                positive_class_weight=1.0,
                focal_gamma=1.0,
                patience=2,
                device="cpu",
                mixed_precision=False,
                calibration=True,
                decision_threshold=0.5,
            ),
        )

        summary = run_training(config)
        metrics_path = Path(config.output_dir) / "metrics.json"
        predictions_path = Path(config.output_dir) / "test_predictions.csv"
        queue_path = Path(config.output_dir) / "hitl_queue.csv"

        self.assertTrue(Path(summary["checkpoint_path"]).exists())
        self.assertTrue(metrics_path.exists())
        self.assertTrue(predictions_path.exists())

        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        self.assertIn("test_metrics", metrics)

        evaluation = evaluate_checkpoint(config, summary["checkpoint_path"], split="test")
        self.assertEqual(evaluation["split"], "test")
        self.assertIn("metrics", evaluation)

        build_hitl_queue(predictions_path=predictions_path, output_path=queue_path, top_k=3)
        self.assertTrue(queue_path.exists())

    def test_figures_smoke(self) -> None:
        rng = np.random.default_rng(0)
        n = 40
        logits = rng.normal(0.0, 2.0, size=n).astype(np.float32)
        targets = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.float32)
        temperature = 1.5

        results = {
            "ModelA": {
                "logits": logits,
                "targets": targets,
                "temperature": temperature,
                "metrics": {
                    "precision_at_fpr_1e-2": 0.8,
                    "precision_at_fpr_1e-3": 0.6,
                    "recall_at_top_10": 0.9,
                    "recall_at_top_25": 1.0,
                },
            },
            "ModelB": {
                "logits": rng.normal(0.0, 1.5, size=n).astype(np.float32),
                "targets": targets,
                "temperature": None,
                "metrics": {
                    "precision_at_fpr_1e-2": None,
                    "precision_at_fpr_1e-3": None,
                    "recall_at_top_10": None,
                    "recall_at_top_25": None,
                },
            },
        }

        figures_dir = self.test_root / "figures"
        pr_path = figures_dir / "pr_curve.png"
        rel_path = figures_dir / "reliability_diagram.png"

        returned_pr = plot_pr_curve(results, output_path=pr_path)
        self.assertTrue(pr_path.exists(), "PR curve PNG was not created")
        self.assertEqual(returned_pr, pr_path)

        returned_rel = plot_reliability_diagram(results, output_path=rel_path, n_bins=5)
        self.assertTrue(rel_path.exists(), "Reliability diagram PNG was not created")
        self.assertEqual(returned_rel, rel_path)

        table = format_low_fpr_table(results)
        self.assertIn("ModelA", table)
        self.assertIn("ModelB", table)
        self.assertIn("P@FPR=1%", table)
        self.assertIn("n/a", table)
        # ModelA has a valid value
        self.assertIn("0.8000", table)

if __name__ == "__main__":
    unittest.main()
