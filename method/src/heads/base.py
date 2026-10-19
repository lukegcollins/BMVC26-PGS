"""
Abstract Head Interface for DeepfakeWorkbench.

ARCHITECTURE PHILOSOPHY:
=======================
Heads are TASK-SPECIFIC LAYERS that sit on top of Backbones. They transform
generic feature representations into task-specific outputs.

Heads do NOT:
  - Extract features from raw images (that's the Backbone's job)
  - Compute losses (that's the Loss's job)
  - Wire together the pipeline (that's the Detector's job)

Heads DO:
  - Transform backbone features into task outputs
  - Handle task-specific architectures (classifiers, projections, etc.)
  - Provide dimensionality reduction/expansion as needed
  - Support various head types: linear, MLP, contrastive projection, etc.

Examples:
  - Classification Head: features -> logits [B, D] -> [B, num_classes]
  - Contrastive Projection Head: features -> embeddings [B, D] -> [B, embed_dim]
  - Temporal Head: sequence features -> video-level features [B, T, D] -> [B, D']
  - Multi-task Head: features -> multiple outputs (classification + regression)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.configs import HeadConfig


@dataclass
class HeadOutput:
    """
    Standard output structure from head forward pass.

    Attributes:
        output: Main output tensor (e.g., logits, embeddings)
        auxiliary: Optional dict of auxiliary outputs for multi-task learning
        intermediate: Optional dict of intermediate layer outputs
    """

    output: torch.Tensor
    auxiliary: dict[str, torch.Tensor] | None = None
    intermediate: dict[str, torch.Tensor] | None = None


class BaseHead(nn.Module, ABC):
    """
    Abstract base class for all task-specific heads.

    DESIGN CONTRACT:
    ================
    Heads are the second stage of the detection pipeline. They receive
    feature representations from Backbones and produce task-specific outputs.

    Required Interface:
    -------------------
    1. forward(features) -> HeadOutput
       - Input: Feature tensor from backbone [B, D] or [B, T, D]
       - Output: HeadOutput with task-specific .output

    2. get_output_dim() -> int
       - Returns output dimension (e.g., num_classes or embed_dim)

    Head Types:
    -----------
    - **ClassificationHead**: Linear layer(s) for binary/multi-class prediction
    - **ProjectionHead**: MLP for contrastive learning (e.g., OrthoPairGTL)
    - **TemporalHead**: Process sequences (Transformer, LSTM, Conv1D)
    - **MultiTaskHead**: Multiple outputs (classification + embedding)

    Typical Usage in a Detector:
    -----------------------------
    ```python
    backbone = build_backbone(config.backbone)
    head = build_head(config.head, input_dim=backbone.get_output_dim())

    # In forward pass:
    features = backbone(x).features      # [B, D]
    head_out = head(features)            # [B, num_classes] or [B, embed_dim]
    logits = head_out.output
    ```

    Example Implementations:
    ------------------------
    ```python
    # Simple Linear Classifier
    class LinearHead(BaseHead):
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, features: torch.Tensor) -> HeadOutput:
            logits = self.fc(features)
            return HeadOutput(output=logits)

        def get_output_dim(self) -> int:
            return self.fc.out_features

    # Contrastive Projection Head (for OrthoPairGTL)
    class ProjectionHead(BaseHead):
        def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 2048):
            super().__init__()
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim)
            )

        def forward(self, features: torch.Tensor) -> HeadOutput:
            embeddings = self.projector(features)
            # L2 normalize for contrastive learning
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            return HeadOutput(output=embeddings)

        def get_output_dim(self) -> int:
            return self.projector[-1].out_features

    # Temporal Transformer Head
    class TemporalTransformerHead(BaseHead):
        def __init__(self, input_dim: int, num_heads: int = 8, num_layers: int = 2):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.pooling = nn.AdaptiveAvgPool1d(1)  # Pool over temporal dim

        def forward(self, features: torch.Tensor) -> HeadOutput:
            # features: [B, T, D]
            encoded = self.transformer(features)  # [B, T, D]
            # Pool temporal dimension: [B, T, D] -> [B, D, T] -> [B, D, 1] -> [B, D]
            pooled = self.pooling(encoded.transpose(1, 2)).squeeze(-1)
            return HeadOutput(output=pooled)

        def get_output_dim(self) -> int:
            return self.transformer.layers[0].linear1.in_features
    ```

    Notes for OrthoPairGTL Research:
    ---------------------------------
    For your geometric boundary artifact detection, you'll want a head that:
    1. Can handle paired inputs (center crop vs outer squares)
    2. Projects to a contrastive embedding space
    3. Maintains geometric relationships in feature space

    Consider a specialized head like:
    ```python
    class GeometricContrastiveHead(BaseHead):
        def __init__(self, input_dim: int, embed_dim: int):
            super().__init__()
            self.center_proj = ProjectionHead(input_dim, embed_dim)
            self.outer_proj = ProjectionHead(input_dim, embed_dim)
            # Shared or separate projections depending on research hypothesis
    ```
    """

    def __init__(self, config: HeadConfig | None = None):
        """
        Initialize head.

        Args:
            config: Optional HeadConfig instance with architecture settings
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, features: torch.Tensor, **kwargs) -> HeadOutput:
        """
        Transform features into task-specific output.

        Args:
            features: Feature tensor from backbone [B, D] or [B, T, D]
            **kwargs: Additional arguments for specialized heads

        Returns:
            HeadOutput with task-specific output
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """
        Get the output dimension.

        Returns:
            Output dimension (e.g., num_classes or embed_dim)
        """
        pass

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_groups(self, lr: float, weight_decay: float = 0.0) -> list[dict[str, Any]]:
        """
        Get parameter groups for optimizer.

        Args:
            lr: Learning rate for head parameters
            weight_decay: Weight decay for regularization

        Returns:
            List of parameter group dicts for optimizer
        """
        return [
            {
                "params": self.get_trainable_params(),
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]

    def extra_repr(self) -> str:
        """String representation for printing."""
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        total = sum(1 for _ in self.parameters())
        return f"trainable={trainable}/{total}, output_dim={self.get_output_dim()}"
