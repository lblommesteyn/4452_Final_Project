from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lensing.config import ExperimentConfig
from scripts.run_ablation import AblationSpec, _apply_overrides, _build_specs, _prepare_config


class AblationScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="ablation_test_"))

    def tearDown(self) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_apply_overrides_updates_nested_sections(self) -> None:
        config = ExperimentConfig()
        overrides = {
            "experiment": {"name": "override_name"},
            "data": {"use_weighted_sampler": False},
            "training": {"calibration": False, "positive_class_weight": 5.0},
        }
        _apply_overrides(config, overrides)
        self.assertEqual(config.name, "override_name")
        self.assertFalse(config.data.use_weighted_sampler)
        self.assertFalse(config.training.calibration)
        self.assertEqual(config.training.positive_class_weight, 5.0)

    def test_build_specs_contains_expected_labels(self) -> None:
        args = argparse.Namespace(
            resnet_config=str(ROOT / "configs" / "resnet18_default.yaml"),
            vit_config=str(ROOT / "configs" / "vit_b16_default.yaml"),
        )
        specs = _build_specs(args)
        labels = {spec.label for spec in specs}
        self.assertSetEqual(
            labels,
            {
                "resnet_weighted_calibrated",
                "resnet_weighted_raw",
                "resnet_weighted_bce",
                "resnet_unweighted_bce",
                "resnet_unweighted_focal",
                "vit_weighted_calibrated",
            },
        )
        for spec in specs:
            if spec.label.startswith("resnet"):
                self.assertTrue(spec.config_path.name.startswith("resnet"))
            if spec.label.startswith("vit"):
                self.assertTrue(spec.config_path.name.startswith("vit"))

    def test_prepare_config_sets_unique_output_dir(self) -> None:
        spec = AblationSpec(
            label="resnet_weighted_calibrated",
            description="",
            config_path=Path(ROOT / "configs" / "resnet18_default.yaml"),
            overrides={"training": {"calibration": False}},
        )
        config = _prepare_config(spec, self.tmp_dir)
        self.assertEqual(config.name, spec.label)
        self.assertEqual(Path(config.output_dir), self.tmp_dir / spec.label)
        self.assertFalse(config.training.calibration)


if __name__ == "__main__":
    unittest.main()
