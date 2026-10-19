"""
Loss functions for DeepfakeWorkbench.

This module provides:
- Abstract loss interface for consistent API
- Standard classification losses (BCE, CE)
- Contrastive losses (OrthoPair, GTL, PairwiseContrastive)
- OrthoPairGTL (geometric transformation learning with orthogonal alignment)
- VideoPrism losses (SupCon, TemporalDerivative, VideoPrismCombined)
- Combined losses with dynamic weighting
- Loss registry for dynamic instantiation
"""

from src.loss.abstract_loss import BaseLoss, LossOutput
from src.loss.gsd_set_loss import GSDSetLoss
from src.loss.orthoflow_gtl import OrthoPairGTLLoss
from src.loss.registry import build_loss, loss_registry

__all__ = [
    "BaseLoss",
    "LossOutput",
    "loss_registry",
    "build_loss",
    "GSDSetLoss",
    "OrthoPairGTLLoss",
]
