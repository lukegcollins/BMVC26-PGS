"""
Abstract Detector Interface for DeepfakeWorkbench.

ARCHITECTURE PHILOSOPHY:
=======================
Detectors are THE ORCHESTRATORS. They wire together the three core components
of a detection pipeline:

  1. Backbone (src.backbones) - Extracts features from raw images
  2. Head (src.heads) - Transforms features into task-specific outputs
  3. Loss (src.loss) - Computed externally, but detector provides the right outputs

Detectors do NOT:
  - Extract features directly (delegates to Backbone)
  - Define task-specific layers (delegates to Head)
  - Compute losses (that's handled by LightningModule or Loss classes)

Detectors DO:
  - Wire forward() pass: x -> Backbone -> Head -> DetectorOutput
  - Manage component initialization from configs
  - Provide unified interfaces (get_features, get_param_groups, etc.)
  - Handle special cases (temporal, paired, multi-task, etc.)

The detector is the "glue" that makes components work together seamlessly.
It ensures Backbone outputs feed correctly into Head inputs, and that the
final output format matches what the Loss expects.

Example Detection Flow:
=======================
Input Image [B, C, H, W]
    ↓
Backbone (DINOv2, ResNet, etc.)
    ↓
Features [B, D]
    ↓
Head (Linear, MLP, Projection, etc.)
    ↓
Output [B, num_classes] or [B, embed_dim]
    ↓
DetectorOutput (standardized format)
    ↓
Loss Function (computes training signal)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from src.configs import ExperimentConfig


@dataclass
class DetectorOutput:
    """
    Standard output structure from detector forward pass.

    Attributes:
        logits: Classification logits [B, num_classes]
        probabilities: Softmax probabilities [B, num_classes]
        features: Backbone features before classification [B, D]
        embeddings: Optional intermediate embeddings for contrastive learning
        auxiliary: Optional dict with auxiliary outputs (for multi-task losses)
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    features: torch.Tensor
    embeddings: torch.Tensor | None = None
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)


class BaseDetector(nn.Module, ABC):
    """
    Abstract base class for all deepfake detectors.

    DESIGN CONTRACT:
    ================
    Detectors are the ORCHESTRATORS that wire together modular components
    into a complete detection pipeline.

    Required Interface:
    -------------------
    1. forward(x, **kwargs) -> DetectorOutput
       - Orchestrates: x -> Backbone -> Head -> DetectorOutput
       - Returns standardized output with logits, probabilities, features

    2. get_features(x) -> torch.Tensor
       - Extract raw features without classification
       - Useful for visualization, similarity, downstream tasks

    3. get_param_groups(lr, ...) -> list[dict]
       - Returns optimizer parameter groups
       - Allows differential learning rates for backbone vs head

    Component Wiring:
    -----------------
    Detectors compose these pluggable components:

    ```
    self.backbone = build_backbone(config.backbone)  # Feature extractor
    self.head = build_head(config.head, ...)         # Task-specific layer
    ```

    The forward pass wires them together:

    ```
    def forward(self, x):
        backbone_out = self.backbone(x)        # [B,C,H,W] -> BackboneOutput
        features = backbone_out.features       # [B, D]
        head_out = self.head(features)         # [B, D] -> HeadOutput
        logits = head_out.output               # [B, num_classes]
        return DetectorOutput(...)
    ```

    Detector Types:
    ---------------
    - **StandardDetector**: Single image -> classification
    - **TemporalDetector**: Video sequence -> classification
    - **PairedDetector**: Image pairs -> contrastive embeddings
    - **GeometricDetector**: Multi-region (center vs boundary) -> OrthoPairGTL

    Example Implementation:
    -----------------------
    ```python
    from src.backbones import build_backbone
    from src.heads import build_head

    class StandardDetector(BaseDetector):
        '''Simple image classification detector.'''

        def __init__(self, config: ExperimentConfig):
            super().__init__(config)

            # Build components from registry
            self.backbone = build_backbone(config.backbone)
            self.head = build_head(
                config.head,
                input_dim=self.backbone.get_output_dim(),
                num_classes=2  # real/fake
            )

        def forward(self, x: torch.Tensor, **kwargs) -> DetectorOutput:
            # 1. Extract features via backbone
            backbone_out = self.backbone(x)  # [B, C, H, W] -> [B, D]

            # 2. Get task output via head
            head_out = self.head(backbone_out.features)  # [B, D] -> [B, 2]

            # 3. Return standardized output
            logits = head_out.output
            return DetectorOutput(
                logits=logits,
                probabilities=F.softmax(logits, dim=-1),
                features=backbone_out.features
            )

        def get_features(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x).features

        def get_param_groups(self, lr, lr_backbone_mult=0.1, weight_decay=1e-4):
            return [
                {
                    "params": self.backbone.parameters(),
                    "lr": lr * lr_backbone_mult,
                    "weight_decay": weight_decay
                },
                {
                    "params": self.head.parameters(),
                    "lr": lr,
                    "weight_decay": weight_decay
                }
            ]
    ```

    For OrthoPairGTL Geometric Detection:
    -------------------------------------
    Your geometric boundary artifact detection will use a specialized detector:

    ```python
    class GeometricDetector(BaseDetector):
        '''Compares face center to outer boundary regions.'''

        def __init__(self, config: ExperimentConfig):
            super().__init__(config)
            self.backbone = build_backbone(config.backbone)  # Shared
            self.center_head = build_head(config.head, ...)  # Center features
            self.outer_head = build_head(config.head, ...)   # Boundary features
            # Heads project to contrastive embedding space

        def forward(self, x: torch.Tensor, **kwargs) -> DetectorOutput:
            # Extract center and outer regions (using Kornia)
            center_crop = extract_center(x)
            outer_squares = extract_boundary(x)

            # Shared backbone
            center_feat = self.backbone(center_crop).features
            outer_feat = self.backbone(outer_squares).features

            # Separate heads for geometric contrast
            center_embed = self.center_head(center_feat).output
            outer_embed = self.outer_head(outer_feat).output

            # Return embeddings for OrthoPairGTL loss
            return DetectorOutput(
                logits=None,  # No direct classification
                embeddings={"center": center_embed, "outer": outer_embed},
                ...
            )
    ```
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize detector.

        Args:
            config: ExperimentConfig with backbone and head settings
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        **kwargs,
    ) -> DetectorOutput:
        """
        Forward pass for classification.

        Args:
            x: Input tensor [B, C, H, W] for images or [B, T, C, H, W] for sequences
            return_features: If True, include intermediate features in output
            **kwargs: Additional arguments (e.g., pair_x for contrastive learning)

        Returns:
            DetectorOutput with classification results
        """
        pass

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.

        Useful for:
        - Feature visualization
        - Similarity computation
        - Downstream tasks

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Feature tensor [B, D]
        """
        pass

    @abstractmethod
    def get_param_groups(
        self,
        lr: float,
        lr_backbone_mult: float = 0.1,
        weight_decay: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """
        Get parameter groups for optimizer.

        Typically returns separate groups for:
        - Backbone parameters (possibly with lower LR)
        - Head parameters (full LR)

        Args:
            lr: Base learning rate
            lr_backbone_mult: Multiplier for backbone LR
            weight_decay: Weight decay value

        Returns:
            List of parameter group dicts for optimizer
        """
        pass

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = True

    def get_num_params(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str) -> None:
        """
        Save model weights and config.

        Args:
            path: Directory path to save to
        """
        import json
        from dataclasses import asdict
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), save_dir / "model.pt")

        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

    @classmethod
    def from_pretrained(cls, path: str) -> "BaseDetector":
        """
        Load model from saved weights and config.

        Args:
            path: Directory path to load from

        Returns:
            Loaded detector instance
        """
        import json
        from pathlib import Path

        from src.configs import ExperimentConfig

        load_dir = Path(path)

        # Load config
        with open(load_dir / "config.json") as f:
            config_dict = json.load(f)

        config = ExperimentConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load weights
        state_dict = torch.load(load_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        return model
