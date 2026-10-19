"""
Deepfake detection models for DeepfakeWorkbench.

This module provides:
- Abstract detector interface for consistent API
- Classification head implementations (Linear, MLP, Attention)
- Temporal heads for sequence/video-level detection
- Complete detector models combining backbone + head
- Detector registry for dynamic instantiation
"""

from src.detectors.base_detector import BaseDetector, DetectorOutput
from src.detectors.orthoflow_detector import OrthoFlowDetector
from src.detectors.registry import build_detector, detector_registry

__all__ = [
    "BaseDetector",
    "DetectorOutput",
    "detector_registry",
    "build_detector",
    "OrthoFlowDetector",
]
