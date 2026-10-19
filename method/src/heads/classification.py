"""
Classification Head implementations for DeepfakeWorkbench.

Migrated from the inline head construction in the original OrthoFlowGTLDetector.
Each head conforms to the BaseHead interface and is registered with the head_registry.
"""

import logging

import torch
import torch.nn as nn

from src.configs import HeadConfig
from src.heads.base import BaseHead, HeadOutput

logger = logging.getLogger(__name__)


def _init_weights(module: nn.Module) -> None:
    """Initialize head weights with truncated normal and zero biases."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class LinearHead(BaseHead):
    """
    Simple linear classifier: Dropout -> Linear.

    Config keys (via HeadConfig):
        - num_classes: int (default 2)
        - dropout: float (default 0.1)
    """

    def __init__(self, config: HeadConfig | None = None, input_dim: int = 1024, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(config)

        if config is not None:
            num_classes = config.num_classes
            dropout = config.dropout

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )
        _init_weights(self.head)

        logger.info(f"LinearHead: {input_dim} -> {num_classes} (dropout={dropout})")

    def forward(self, features: torch.Tensor, **kwargs) -> HeadOutput:
        logits = self.head(features)
        return HeadOutput(output=logits)

    def get_output_dim(self) -> int:
        return self.num_classes


class MLPHead(BaseHead):
    """
    Two-layer MLP classifier: Linear -> GELU -> Dropout -> Linear.

    Config keys (via HeadConfig):
        - hidden_dim: int (default 512)
        - num_classes: int (default 2)
        - dropout: float (default 0.1)
    """

    def __init__(
        self,
        config: HeadConfig | None = None,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(config)

        if config is not None:
            hidden_dim = config.hidden_dim
            num_classes = config.num_classes
            dropout = config.dropout

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        _init_weights(self.head)

        logger.info(f"MLPHead: {input_dim} -> {hidden_dim} -> {num_classes} (dropout={dropout})")

    def forward(self, features: torch.Tensor, **kwargs) -> HeadOutput:
        logits = self.head(features)
        return HeadOutput(output=logits)

    def get_output_dim(self) -> int:
        return self.num_classes


class MLP3LayerHead(BaseHead):
    """
    Three-layer MLP: Linear -> GELU -> Drop -> Linear -> GELU -> Drop -> Linear.

    Config keys (via HeadConfig.extra):
        - hidden_dim: int (default 512)
        - num_classes: int (default 2)
        - dropout: float (default 0.1)
    """

    def __init__(
        self,
        config: HeadConfig | None = None,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(config)

        if config is not None:
            hidden_dim = config.hidden_dim
            num_classes = config.num_classes
            dropout = config.dropout

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        _init_weights(self.head)

        logger.info(f"MLP3LayerHead: {input_dim} -> {hidden_dim} -> {hidden_dim // 2} -> {num_classes}")

    def forward(self, features: torch.Tensor, **kwargs) -> HeadOutput:
        logits = self.head(features)
        return HeadOutput(output=logits)

    def get_output_dim(self) -> int:
        return self.num_classes


class MLPBatchNormHead(BaseHead):
    """
    MLP with BatchNorm: Linear -> BN -> GELU -> Linear.

    Config keys (via HeadConfig):
        - hidden_dim: int (default 512)
        - num_classes: int (default 2)
    """

    def __init__(
        self,
        config: HeadConfig | None = None,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 2,
    ):
        super().__init__(config)

        if config is not None:
            hidden_dim = config.hidden_dim
            num_classes = config.num_classes

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        _init_weights(self.head)

        logger.info(f"MLPBatchNormHead: {input_dim} -> BN({hidden_dim}) -> {num_classes}")

    def forward(self, features: torch.Tensor, **kwargs) -> HeadOutput:
        logits = self.head(features)
        return HeadOutput(output=logits)

    def get_output_dim(self) -> int:
        return self.num_classes
