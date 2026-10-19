"""
Configuration management for DeepfakeWorkbench training framework.

This module provides:
- Dataclass-based configuration objects for type-safe config handling
- YAML configuration loading and validation
- Configuration merging and inheritance
"""

from src.configs.base_config import (
    AugmentationConfig,
    BackboneConfig,
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    HeadConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    load_config,
    save_config,
)

__all__ = [
    "ExperimentConfig",
    "BackboneConfig",
    "HeadConfig",
    "LossConfig",
    "DataConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CheckpointConfig",
    "AugmentationConfig",
    "load_config",
    "save_config",
]
