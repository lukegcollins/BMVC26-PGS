"""
Dataset Factory for DeepfakeWorkbench.

This module provides factory functions for instantiating datasets and transforms
from configuration dictionaries.
"""

from src.dataset.components.dual_paired import DualPairSequenceDataset
from src.dataset.components.paired import PairedFrameDataset, PairedSequenceDataset
from src.dataset.components.single import StandardFrameDataset, StandardSequenceDataset
from src.dataset.transforms import build_transform

# Dataset registry
DATASETS = {
    "StandardFrameDataset": StandardFrameDataset,
    "StandardSequenceDataset": StandardSequenceDataset,
    "PairedFrameDataset": PairedFrameDataset,
    "PairedSequenceDataset": PairedSequenceDataset,
    "DualPairSequenceDataset": DualPairSequenceDataset,
}


def build_dataset(config: dict, split: str = "train"):
    """
    Factory function to instantiate the correct dataset.

    Args:
        config: The 'dataset' section of your experiment yaml.
            Required keys:
                - name: Dataset class name (e.g., "StandardFrameDataset")
                - params: Dict with dataset parameters
                    - manifest_path: Path to JSON manifest
                    - image_size: Optional, default 224
                    - transforms: Optional, list of transform configs
        split: 'train', 'val', or 'test'.

    Returns:
        Dataset instance

    Raises:
        ValueError: If manifest_path not provided or unknown dataset class

    Example:
        config = {
            "name": "StandardFrameDataset",
            "params": {
                "manifest_path": "path/to/manifest.json",
                "image_size": 224,
                "scan_mode": "balanced",
                "max_frames_per_video": 50,
            }
        }
        dataset = build_dataset(config, split="train")
    """
    name = config.get("name")
    params = config.get("params", {}).copy()  # Copy to avoid mutation
    manifest_path = params.pop("manifest_path", None)

    if not manifest_path:
        raise ValueError("manifest_path must be provided in dataset params")

    # Build transform (supports both simple and config-driven modes)
    transform = build_transform(params, is_training=(split == "train"))

    dataset_class = DATASETS.get(name)
    if not dataset_class:
        available = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(f"Unknown dataset class: '{name}'. Available: [{available}]")

    # Inject split and transform
    return dataset_class(manifest_path=manifest_path, split=split, transform=transform, **params)


def register_dataset(name: str, cls: type) -> None:
    """
    Register a custom dataset class.

    Args:
        name: Name to register under
        cls: Dataset class (should inherit from DeepfakeBaseDataset)

    Example:
        from src.dataset.core.base import DeepfakeBaseDataset

        class MyCustomDataset(DeepfakeBaseDataset):
            def __getitem__(self, idx):
                ...

        register_dataset("MyCustomDataset", MyCustomDataset)
    """
    DATASETS[name] = cls


__all__ = ["build_dataset", "build_transform", "DATASETS", "register_dataset"]
