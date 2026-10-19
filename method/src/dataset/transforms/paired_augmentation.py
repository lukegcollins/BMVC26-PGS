"""
Paired augmentation for VideoPrism temporal deepfake detection.

Implements the augmentation policy from the VideoPrism spec:
- **Shared geometric augmentations**: Same random params applied to both
  real and fake clips in a pair (HorizontalFlip, small Affine).
  Uses Albumentations ReplayCompose so the identical transform is replayed.
- **Independent photometric augmentations**: Different random params per
  clip (JPEG compression, blur, color jitter, noise).

This prevents the model from learning identity-based shortcuts while
allowing it to learn compression/noise-based artifact differences.

Usage:
    aug = PairedAugmentation(image_size=256)
    real_aug, fake_aug = aug(real_image, fake_image)

    # Or build from YAML config dict:
    aug = build_paired_augmentation({
        "image_size": 256,
        "geometric": [
            {"name": "HorizontalFlip", "params": {"p": 0.5}},
            {"name": "Affine", "params": {"scale": [0.95, 1.05], "rotate": [-5, 5], "p": 0.3}},
        ],
        "photometric": [
            {"name": "ImageCompression", "params": {"quality_lower": 60, "quality_upper": 100, "p": 0.3}},
            {"name": "GaussianBlur", "params": {"blur_limit": [3, 5], "p": 0.2}},
        ],
    })
"""

from __future__ import annotations

import logging
from typing import Any

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from src.dataset.transforms.builder import TRANSFORM_REGISTRY

logger = logging.getLogger(__name__)

# Default geometric transforms (shared across pair)
DEFAULT_GEOMETRIC = [
    {"name": "HorizontalFlip", "params": {"p": 0.5}},
    {"name": "Affine", "params": {"scale": [0.95, 1.05], "rotate": [-5, 5], "p": 0.3}},
]

# Default photometric transforms (independent per clip)
DEFAULT_PHOTOMETRIC = [
    {"name": "ImageCompression", "params": {"quality_lower": 60, "quality_upper": 100, "p": 0.3}},
    {"name": "GaussianBlur", "params": {"blur_limit": [3, 5], "p": 0.2}},
    {"name": "ColorJitter", "params": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05, "p": 0.3}},
    {"name": "GaussNoise", "params": {"var_limit": [5.0, 30.0], "p": 0.2}},
]

# Default normalization
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]


def _build_transforms(config_list: list[dict[str, Any]]) -> list[A.BasicTransform]:
    """Build a list of Albumentations transform instances from config dicts."""
    transforms = []
    for item in config_list:
        name = item.get("name")
        params = item.get("params", {})

        if name not in TRANSFORM_REGISTRY:
            raise ValueError(
                f"Unknown transform '{name}'. "
                f"Available: {sorted(TRANSFORM_REGISTRY.keys())}"
            )

        cls = TRANSFORM_REGISTRY[name]
        try:
            transforms.append(cls(**params))
        except Exception as e:
            raise ValueError(f"Failed to create transform {name}({params}): {e}") from e

    return transforms


class PairedAugmentation:
    """
    Paired augmentation pipeline for temporally aligned real/fake clips.

    Applies:
    1. Resize to target size
    2. Shared geometric augmentations (identical params for both images)
    3. Independent photometric augmentations (different params per image)
    4. Normalize + ToTensor

    The shared geometric stage uses ``A.ReplayCompose`` so the exact same
    random transform is applied to both images in a pair.

    Args:
        image_size: Target image size (height=width).
        geometric_config: List of transform config dicts for shared geometric augs.
            If None, uses DEFAULT_GEOMETRIC.
        photometric_config: List of transform config dicts for independent photometric augs.
            If None, uses DEFAULT_PHOTOMETRIC.
        normalize_mean: Normalize mean. Default ImageNet values.
        normalize_std: Normalize std. Default ImageNet values.
        to_tensor: Whether to append ToTensorV2. Default True.

    Example:
        aug = PairedAugmentation(image_size=256)
        real_tensor, fake_tensor = aug(real_np, fake_np)
    