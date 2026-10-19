"""
Abstract loss interface for DeepfakeWorkbench.

All loss implementations must inherit from BaseLoss and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.configs import LossConfig


@dataclass
class LossOutput:
    """
    Standard output structure from loss computation.

    Attributes:
        total: Total weighted loss (scalar)
        components: Dict of individual loss components with values
        weights: Dict of weights applied to each component
        metrics: Optional dict of additional metrics to log
    """

    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class BaseLoss(nn.Module, ABC):
    """
    Abstract base class for all loss functions.

    Loss functions compute training objectives from model outputs.
    All losses must:
    1. Implement forward() returning LossOutput
    2. Support configurable weighting
    3. Track component losses for monitoring

    Example:
        class MyLoss(BaseLoss):
            def __init__(self, config: LossConfig):
                super().__init__(config)
                self.ce = nn.CrossEntropyLoss()

            def forward(self, outputs, targets, **kwargs) -> LossOutput:
                ce_loss = self.ce(outputs.logits, targets)
                return LossOutput(
                    total=ce_loss,
                    components={"ce": ce_loss},
                    weights={"ce": 1.0},
                )
    """

    def __init__(self, config: LossConfig):
        """
        Initialize loss.

        Args:
            config: LossConfig instance with loss settings
        """
        super().__init__()
        self.config = config
        self._step = 0

    @abstractmethod
    def forward(
        self,
        outputs,
        targets: torch.Tensor,
        **kwargs,
    ) -> LossOutput:
        """
        Compute loss from model outputs and targets.

        Args:
            outputs: DetectorOutput from model forward pass
            targets: Ground truth labels [B]
            **kwargs: Additional arguments (e.g., pair data for contrastive losses)

        Returns:
            LossOutput with total loss and components
        """
        pass

    def step(self) -> None:
        """Increment internal step counter (for scheduled weights)."""
        self._step += 1

    def get_current_weights(self) -> dict[str, float]:
        """
        Get current loss weights (may be dynamic).

        Returns:
            Dict of component name to weight
        """
        return {
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "gamma": self.config.gamma,
        }

    def set_eval_mode(self, eval_mode: bool = True) -> None:
        """
        Set evaluation mode for loss computation.

        Some losses behave differently during evaluation
        (e.g., only compute CE loss, disable contrastive losses).

        Args:
            eval_mode: If True, enable eval mode
        """
        self._eval_mode = eval_mode
