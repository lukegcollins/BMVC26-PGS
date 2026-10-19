"""
Neural network backbones for DeepfakeWorkbench.

This module provides:
- Abstract backbone interface for consistent API
- Pre-trained vision transformer backbones (DINOv2, DINOv3)
- Backbone registry for dynamic instantiation
"""

from src.backbones.base import BackboneOutput, BaseBackbone
from src.backbones.registry import backbone_registry, build_backbone

__all__ = [
    "BaseBackbone",
    "BackboneOutput",
    "backbone_registry",
    "build_backbone",
]
