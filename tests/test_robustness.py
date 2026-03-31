# Test the code for rubustness.py 
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lensing.robustness import (
    add_gaussian_noise,
    apply_blur,
    adjust_contrast,
    shift_images,
    apply_perturbation,
)


class TestRobustness(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.rand(2, 3, 32, 32)

    def test_noise_preserves_shape(self) -> None:
        y = add_gaussian_noise(self.x, std=0.05)
        self.assertEqual(y.shape, self.x.shape)

    def test_blur_preserves_shape(self) -> None:
        y = apply_blur(self.x, kernel_size=3)
        self.assertEqual(y.shape, self.x.shape)

    def test_contrast_preserves_shape(self) -> None:
        y = adjust_contrast(self.x, factor=1.2)
        self.assertEqual(y.shape, self.x.shape)

    def test_shift_preserves_shape(self) -> None:
        y = shift_images(self.x, shift_x=2, shift_y=2)
        self.assertEqual(y.shape, self.x.shape)

    def test_zero_noise_is_identity(self) -> None:
        y = add_gaussian_noise(self.x, std=0.0)
        self.assertTrue(torch.allclose(y, self.x))

    def test_blur_kernel_one_is_identity(self) -> None:
        y = apply_blur(self.x, kernel_size=1)
        self.assertTrue(torch.allclose(y, self.x))

    def test_contrast_factor_one_is_identity(self) -> None:
        y = adjust_contrast(self.x, factor=1.0)
        self.assertTrue(torch.allclose(y, self.x))

    def test_zero_shift_is_identity(self) -> None:
        y = shift_images(self.x, shift_x=0, shift_y=0)
        self.assertTrue(torch.allclose(y, self.x))

    def test_noise_changes_values(self) -> None:
        y = add_gaussian_noise(self.x, std=0.05)
        self.assertFalse(torch.allclose(y, self.x))

    def test_apply_perturbation_clean(self) -> None:
        y = apply_perturbation(self.x, "clean")
        self.assertTrue(torch.allclose(y, self.x))

    def test_apply_perturbation_invalid_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_perturbation(self.x, "bad_name")


if __name__ == "__main__":
    unittest.main()