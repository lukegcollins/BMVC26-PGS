"""
Abstract Backbone Interface for DeepfakeWorkbench.

ARCHITECTURE PHILOSOPHY:
=======================
Backbones are PURE FEATURE EXTRACTORS. They take raw image tensors and output
feature representations. Backbones do NOT:
  - Classify (that's the Head's job)
  - Compute losses (that's the Loss's job)
  - Wire together the pipeline (that's the Detector's job)

Backbones DO:
  - Extract semantic features from images/videos
  - Handle various vision architectures (ViTs, CNNs, hybrids)
  - Support freezing strategies for transfer learning
  - Provide standardized output format (BackboneOutput)

All backbone implementations must inherit from BaseBackbone and implement
the required abstract methods. This ensures plug-and-play compatibility
with any Head and any Detector.

Example architectures: DINOv2, DINOv3, ResNet, EfficientNet, Swin, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.configs import BackboneConfig


@dataclass
class BackboneOutput:
    """
    Standard output structure from backbone forward pass.

    Attributes:
        features: Main feature tensor [B, D] or [B, T, D] for sequences
        cls_token: Class token if available [B, D]
        patch_tokens: Patch tokens if available [B, N, D]
        intermediate: Optional dict of intermediate layer outputs
    """

    features: torch.Tensor
    cls_token: torch.Tensor | None = None
    patch_tokens: torch.Tensor | None = None
    intermediate: dict[str, torch.Tensor] | None = None


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for all backbone feature extractors.

    DESIGN CONTRACT:
    ================
    Backbones are the first stage of the detection pipeline. They receive
    raw image tensors and output semantic feature representations.

    Required Interface:
    -------------------
    1. forward(x) -> BackboneOutput
       - Input: [B, C, H, W] images or [B, T, C, H, W] video sequences
       - Output: BackboneOutput with .features, .cls_token, .patch_tokens

    2. get_output_dim() -> int
       - Returns feature dimension for downstream heads to consume

    Optional Features:
    ------------------
    - Freezing strategies (full, layernorm, partial, none)
    - Parameter grouping for differential learning rates
    - Normalization handling (mean/std from config)

    Typical Usage in a Detector:
    -----------------------------
    ```python
    backbone = build_backbone(config.backbone)  # Factory from registry
    head = build_head(config.head, input_dim=backbone.get_output_dim())

    # In forward pass:
    backbone_out = backbone(x)           # [B, C, H, W] -> BackboneOutput
    features = backbone_out.features     # [B, D]
    logits = head(features)              # [B, num_classes]
    ```

    Example Implementation:
    -----------------------
    ```python
    class DINOv2Backbone(BaseBackbone):
        def __init__(self, config: BackboneConfig):
            super().__init__(config)
            self.model = torch.hub.load('facebookresearch/dinov2', config.name)
            self._apply_freeze()

        def forward(self, x: torch.Tensor) -> BackboneOutput:
            output = self.model(x, is_training=True)
            return BackboneOutput(
                features=output['x_norm_clstoken'],
                cls_token=output['x_norm_clstoken'],
                patch_tokens=output['x_norm_patchtokens']
            )

        def get_output_dim(self) -> int:
            return self.model.embed_dim  # e.g., 1024 for ViT-L
    ```
    """

    def __init__(self, config: BackboneConfig):
        """
        Initialize backbone.

        Args:
            config: BackboneConfig instance with architecture settings
        """
        super().__init__()
        self.config = config
        self._frozen_params: set[str] = set()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """
        Extract features from input tensor.

        Args:
            x: Input tensor [B, C, H, W] for images or [B, T, C, H, W] for sequences

        Returns:
            BackboneOutput with extracted features
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """
        Get the output feature dimension.

        Returns:
            Feature dimension (e.g., 1024 for ViT-L)
        """
        pass

    def get_num_patches(self, input_size: int = 224) -> int:
        """
        Get number of patch tokens for given input size.

        Args:
            input_size: Input image size (assumes square)

        Returns:
            Number of patch tokens (default implementation assumes 14x14 patches)
        """
        patch_size = getattr(self, "patch_size", 14)
        return (input_size // patch_size) ** 2

    def _apply_freeze(self) -> None:
        """
        Apply freezing strategy based on config.freeze_mode.

        Supported modes:
        - 'full': Freeze all parameters
        - 'layernorm': Freeze all except LayerNorm layers
        - 'none': No freezing, all parameters trainable
        - 'partial': Freeze first N layers (requires extra config)
        """
        freeze_mode = self.config.freeze_mode

        if freeze_mode == "none":
            return

        for name, param in self.named_parameters():
            should_freeze = False

            if freeze_mode == "full":
                should_freeze = True
            elif freeze_mode == "layernorm":
                # Freeze everything except LayerNorm
                is_layernorm = "norm" in name.lower() or "ln" in name.lower()
                should_freeze = not is_layernorm
            elif freeze_mode == "partial":
                # Freeze layers below threshold
                layer_threshold = self.config.extra.get("freeze_layers", 12)
                should_freeze = self._is_layer_below_threshold(name, layer_threshold)

            if should_freeze:
                param.requires_grad = False
                self._frozen_params.add(name)

    def _is_layer_below_threshold(self, param_name: str, threshold: int) -> bool:
        """
        Check if parameter belongs to a layer below the threshold.

        Args:
            param_name: Parameter name (e.g., 'blocks.5.attn.qkv.weight')
            threshold: Layer number threshold

        Returns:
            True if layer should be frozen
        """
        import re

        # Extract layer number from parameter name
        match = re.search(r"blocks?\.(\d+)", param_name)
        if match:
            layer_num = int(match.group(1))
            return layer_num < threshold
        # Non-block parameters (embeddings, etc.) are frozen
        return True

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen_params.clear()

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_frozen_params(self) -> list[str]:
        """Get list of frozen parameter names."""
        return list(self._frozen_params)

    def get_param_groups(self, lr: float, lr_backbone_mult: float = 0.1) -> list[dict[str, Any]]:
        """
        Get parameter groups with different learning rates.

        Useful for fine-tuning where backbone uses lower LR than head.

        Args:
            lr: Base learning rate (for non-frozen params)
            lr_backbone_mult: Multiplier for backbone parameters

        Returns:
            List of parameter group dicts for optimizer
        """
        backbone_params = []
        other_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if name in self._frozen_params:
                    continue
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr * lr_backbone_mult},
        ]

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor using config mean/std.

        Args:
            x: Input tensor [B, C, H, W] with values in [0, 1]

        Returns:
            Normalized tensor
        """
        mean = torch.tensor(self.config.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.config.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x - mean) / std

    def extra_repr(self) -> str:
        """String representation for printing."""
        frozen = len(self._frozen_params)
        total = sum(1 for _ in self.parameters())
        trainable = total - frozen
        return f"name={self.config.name}, freeze_mode={self.config.freeze_mode}, trainable={trainable}/{total}"
