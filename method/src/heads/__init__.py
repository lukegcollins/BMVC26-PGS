"""
Head modules for DeepfakeWorkbench.

Heads are task-specific layers that transform backbone features into
task outputs (logits, embeddings, etc.).

Available heads:
- LinearHead: Simple dropout + linear classifier
- MLPHead: Two-layer MLP with GELU activation
- MLP3LayerHead: Three-layer MLP
- MLPBatchNormHead: MLP with BatchNorm

Usage:
    from src.heads import build_head
    head = build_head(config.head, input_dim=backbone.get_output_dim())
    output = head(features)  # HeadOutput
"""

from src.heads.base import BaseHead, HeadOutput
from src.heads.classification import (
    LinearHead,
    MLP3LayerHead,
    MLPBatchNormHead,
    MLPHead,
)
from src.heads.registry import build_head, head_registry, register_head

__all__ = [
    "BaseHead",
    "HeadOutput",
    "LinearHead",
    "MLPHead",
    "MLP3LayerHead",
    "MLPBatchNormHead",
    "head_registry",
    "build_head",
    "register_head",
]
