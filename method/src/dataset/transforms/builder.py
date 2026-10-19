"""
Config-Driven Transform Builder for Albumentations.

This module provides utilities to build Albumentations transform pipelines
from configuration dictionaries or YAML, enabling experiment configuration
without code changes.

Usage:
    # From config dict list
    config = [
        {"name": "Resize", "params": {"height": 224, "width": 224}},
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {"name": "Normalize"},
        {"name": "ToTensorV2"},
    ]
    transform = build_transform_from_config(config)

    # Using default presets
    transform = build_transform({"image_size": 224}, is_training=True)
"""

import logging
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

# Registry of available transforms with their Albumentations classes
TRANSFORM_REGISTRY: dict[str, type] = {
    # Spatial transforms
    "Resize": A.Resize,
    "RandomCrop": A.RandomCrop,
    "CenterCrop": A.CenterCrop,
    "RandomResizedCrop": A.RandomResizedCrop,
    "Pad": A.PadIfNeeded,
    # Flip/Rotate
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "Rotate": A.Rotate,
    "RandomRotate90": A.RandomRotate90,
    "ShiftScaleRotate": A.ShiftScaleRotate,  # Left for legacy, but auto-mapped to Affine
    "Affine": A.Affine,
    # Color transforms
    "ColorJitter": A.ColorJitter,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "HueSaturationValue": A.HueSaturationValue,
    "RGBShift": A.RGBShift,
    "CLAHE": A.CLAHE,
    "ToGray": A.ToGray,
    # Blur/Noise
    "GaussianBlur": A.GaussianBlur,
    "MotionBlur": A.MotionBlur,
    "MedianBlur": A.MedianBlur,
    "GaussNoise": A.GaussNoise,
    "ISONoise": A.ISONoise,
    # Quality degradation (useful for deepfake robustness)
    "ImageCompression": A.ImageCompression,
    "Downscale": A.Downscale,
    # Cutout/Dropout
    "CoarseDropout": A.CoarseDropout,
    "GridDropout": A.GridDropout,
    # Normalization
    "Normalize": A.Normalize,
    # Final conversion
    "ToTensorV2": ToTensorV2,
}

# Default normalization values (ImageNet)
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]


def build_transform_from_config(
    config: list[dict[str, Any]],
    additional_targets: dict[str, str] | None = None,
    use_replay: bool = False,
) -> A.Compose | A.ReplayCompose:
    """
    Build an Albumentations pipeline from a config list.
    """
    transforms = []

    for item in config:
        name = item.get("name")
        # Copy params so we don't mutate the loaded config dictionary
        params = dict(item.get("params", {}))

        # --- Legacy Parameter Compatibility Layer ---
        # Auto-convert deprecated Albumentations syntax to modern API

        if name == "ShiftScaleRotate":
            logger.debug("Auto-converting deprecated ShiftScaleRotate to Affine")
            name = "Affine"
            if "shift_limit" in params:
                v = params.pop("shift_limit")
                params["translate_percent"] = [-v, v] if isinstance(v, (float, int)) else v
            if "scale_limit" in params:
                v = params.pop("scale_limit")
                params["scale"] = [1.0 - v, 1.0 + v] if isinstance(v, (float, int)) else v
            if "rotate_limit" in params:
                v = params.pop("rotate_limit")
                params["rotate"] = [-v, v] if isinstance(v, (float, int)) else v
            # Pop border_mode as it behaves differently in Affine, let it default
            params.pop("border_mode", None)

        elif name == "ImageCompression":
            if "quality_lower" in params and "quality_upper" in params:
                params["quality_range"] = [params.pop("quality_lower"), params.pop("quality_upper")]

        elif name == "CoarseDropout":
            if "max_holes" in params:
                params["num_holes_range"] = [1, params.pop("max_holes")]
            if "max_height" in params:
                params["hole_height_range"] = [1, params.pop("max_height")]
            if "max_width" in params:
                params["hole_width_range"] = [1, params.pop("max_width")]
        # ---------------------------------------------

        if name not in TRANSFORM_REGISTRY:
            available = ", ".join(sorted(TRANSFORM_REGISTRY.keys()))
            raise ValueError(f"Unknown transform: '{name}'. Available transforms: [{available}]")

        cls = TRANSFORM_REGISTRY[name]

        # Handle Normalize defaults
        if name == "Normalize" and not params:
            params = {"mean": DEFAULT_NORMALIZE_MEAN, "std": DEFAULT_NORMALIZE_STD}

        try:
            transform = cls(**params)
            transforms.append(transform)
            logger.debug(f"Added transform: {name}({params})")
        except Exception as e:
            raise ValueError(f"Failed to create transform {name}: {e}") from e

    if use_replay:
        return A.ReplayCompose(transforms, additional_targets=additional_targets)
    return A.Compose(transforms, additional_targets=additional_targets)


def build_transform(
    config: dict[str, Any],
    is_training: bool = True,
    additional_targets: dict[str, str] | None = None,
    use_replay: bool = False,
) -> A.Compose | A.ReplayCompose:
    """
    Build transform pipeline using preset configurations.
    """
    if "transforms" in config:
        return build_transform_from_config(
            config["transforms"],
            additional_targets=additional_targets or {"image_pair": "image"},
            use_replay=use_replay,
        )

    size = config.get("image_size", 224)

    if is_training:
        transform_config = [
            {"name": "Resize", "params": {"height": size, "width": size}},
            {"name": "HorizontalFlip", "params": {"p": 0.5}},
            {"name": "Rotate", "params": {"limit": 15, "p": 0.5}},
            {"name": "ColorJitter", "params": {"brightness": 0.2, "contrast": 0.2, "p": 0.2}},
            {"name": "Normalize", "params": {"mean": DEFAULT_NORMALIZE_MEAN, "std": DEFAULT_NORMALIZE_STD}},
            {"name": "ToTensorV2"},
        ]
    else:
        transform_config = [
            {"name": "Resize", "params": {"height": size, "width": size}},
            {"name": "Normalize", "params": {"mean": DEFAULT_NORMALIZE_MEAN, "std": DEFAULT_NORMALIZE_STD}},
            {"name": "ToTensorV2"},
        ]

    return build_transform_from_config(
        transform_config,
        additional_targets=additional_targets or {"image_pair": "image"},
        use_replay=use_replay,
    )


def register_transform(name: str, cls: type) -> None:
    TRANSFORM_REGISTRY[name] = cls
    logger.info(f"Registered custom transform: {name}")
