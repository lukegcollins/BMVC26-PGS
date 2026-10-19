"""
Dataset module for DeepfakeWorkbench.

Provides:
- Abstract base dataset class
- Standard and paired dataset implementations
- Transform builders
- Dataset factory functions
"""

from src.dataset.components.paired import PairedFrameDataset, PairedSequenceDataset
from src.dataset.components.single import StandardFrameDataset, StandardSequenceDataset
from src.dataset.core.base import DeepfakeBaseDataset
from src.dataset.factory import DATASETS, build_dataset, register_dataset
from src.dataset.transforms import build_transform

__all__ = [
    "DeepfakeBaseDataset",
    "StandardFrameDataset",
    "StandardSequenceDataset",
    "PairedFrameDataset",
    "PairedSequenceDataset",
    "build_dataset",
    "build_transform",
    "register_dataset",
    "DATASETS",
]
