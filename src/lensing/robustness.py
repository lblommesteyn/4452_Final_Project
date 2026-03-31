# Contains pure functions for testing the robustness of lensing reconstructions to various effects.
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _validate_images(images: torch.Tensor) -> None:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}")
    if images.numel() == 0:
        raise ValueError("images tensor is empty")


def _clamp_like_input(images: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    ref_min = float(reference.detach().amin().item())
    ref_max = float(reference.detach().amax().item())
    return images.clamp(min=ref_min, max=ref_max)


def add_gaussian_noise(images: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    _validate_images(images)
    if std < 0:
        raise ValueError("std must be non-negative")
    if std == 0:
        return images.clone()

    noisy = images + torch.randn_like(images) * std
    return _clamp_like_input(noisy, images)


def apply_blur(images: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    _validate_images(images)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if kernel_size == 1:
        return images.clone()

    padding = kernel_size // 2
    channels = images.shape[1]

    kernel = torch.ones(
        (channels, 1, kernel_size, kernel_size),
        dtype=images.dtype,
        device=images.device,
    ) / float(kernel_size * kernel_size)

    blurred = F.conv2d(images, kernel, padding=padding, groups=channels)
    return _clamp_like_input(blurred, images)


def adjust_contrast(images: torch.Tensor, factor: float = 1.2) -> torch.Tensor:
    _validate_images(images)
    if factor < 0:
        raise ValueError("factor must be non-negative")
    if factor == 1.0:
        return images.clone()

    mean = images.mean(dim=(2, 3), keepdim=True)
    adjusted = (images - mean) * factor + mean
    return _clamp_like_input(adjusted, images)


def shift_images(images: torch.Tensor, shift_x: int = 2, shift_y: int = 2) -> torch.Tensor:
    _validate_images(images)
    if shift_x == 0 and shift_y == 0:
        return images.clone()

    shifted = torch.roll(images, shifts=(shift_y, shift_x), dims=(2, 3))
    return shifted


def apply_perturbation(
    images: torch.Tensor,
    perturbation: str,
    **kwargs: Any,
) -> torch.Tensor:
    _validate_images(images)

    perturbation = perturbation.lower().strip()

    if perturbation == "clean":
        return images.clone()
    if perturbation == "noise":
        return add_gaussian_noise(images, std=float(kwargs.get("std", 0.05)))
    if perturbation == "blur":
        return apply_blur(images, kernel_size=int(kwargs.get("kernel_size", 3)))
    if perturbation == "contrast":
        return adjust_contrast(images, factor=float(kwargs.get("factor", 1.2)))
    if perturbation == "shift":
        return shift_images(
            images,
            shift_x=int(kwargs.get("shift_x", 2)),
            shift_y=int(kwargs.get("shift_y", 2)),
        )

    valid = {"clean", "noise", "blur", "contrast", "shift"}
    raise ValueError(f"Unknown perturbation '{perturbation}'. Valid options: {sorted(valid)}")