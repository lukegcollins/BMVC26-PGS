"""
OrthoFlowGTL Detector — Refactored for DeepfakeWorkbench architecture.

This detector is the ORCHESTRATOR for the OrthoPairGTL pipeline.
It delegates to:
  - src/backbones/ for feature extraction (DINOv3, DINOv2, etc.)
  - src/heads/ for classification (LinearHead, MLPHead, etc.)
  - src/loss/orthoflow_gtl.py for the loss function (handled externally by LightningModule)

The detector itself contains NO heavy math — it wires the forward pass.

Key design decisions preserved from the paper:
  - L2 normalization of features before the head (config-driven)
  - backbone_force_fp32 option for numerical stability
  - Features returned alongside logits in DetectorOutput for the loss
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones import build_backbone
from src.configs import ExperimentConfig
from src.detectors.base_detector import BaseDetector, DetectorOutput
from src.heads import build_head

logger = logging.getLogger(__name__)


class OrthoFlowDetector(BaseDetector):
    """
    OrthoFlowGTL: Orthogonal Flow with Geodesic Transformation Learning.

    Architecture:
        Image -> Backbone (frozen/PEFT) -> [L2 norm] -> Head -> logits
                                                     |
                                                     +-> features (for loss)

    The loss function (OrthoPairGTLLoss) receives both logits and features
    from DetectorOutput and computes CE + alignment + uniformity + GTL.

    Config keys:
        backbone_name: str - Backbone family ("dinov3", "dinov2", etc.)
        backbone: BackboneConfig - Full backbone configuration
        head: HeadConfig - Classification head configuration
        backbone.extra.backbone_force_fp32: bool - Force FP32 for backbone
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)

        # 1. Build backbone from registry
        self.backbone = build_backbone(config.backbone)
        embed_dim = self.backbone.get_output_dim()

        # 2. Build head from registry
        self.head = build_head(config.head, input_dim=embed_dim)

        # 3. Store config flags
        self.backbone_force_fp32 = config.backbone.extra.get("backbone_force_fp32", False)
        self.l2_normalize = config.backbone.l2_normalize

        if self.backbone_force_fp32:
            logger.warning(
                "backbone_force_fp32=True: Running backbone in FP32. "
                "Consider bfloat16 AMP for better performance."
            )

        # 4. Log model statistics
        self._log_model_info()

    def _log_model_info(self):
        """Log parameter counts for each component."""
        bb_total = sum(p.numel() for p in self.backbone.parameters())
        bb_train = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.head.parameters())
        head_train = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        total = bb_total + head_total
        trainable = bb_train + head_train

        logger.info("=" * 60)
        logger.info("OrthoFlowGTL Detector Statistics")
        logger.info("=" * 60)
        logger.info(
            f"  Backbone: {bb_train:,} / {bb_total:,} trainable "
            f"({100 * bb_train / max(bb_total, 1):.3f}%)"
        )
        logger.info(f"  Head: {head_train:,} / {head_total:,} trainable")
        logger.info(f"  Total: {trainable:,} / {total:,} trainable")
        logger.info("=" * 60)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        pair_x: torch.Tensor | None = None,
        **kwargs,
    ) -> DetectorOutput:
        """
        Forward pass through the detector.

        For paired training (OrthoPairGTL), the LightningModule concatenates
        real and fake images before calling forward. The interleaved labels
        [0, 1, 0, 1, ...] tell the loss which embeddings to pair.

        Args:
            x: Input images [B, C, H, W] or sequences [B, T, C, H, W]
            return_features: Whether to include features in output (always True for training)
            pair_x: Optional paired images (fake counterparts). If provided,
                    concatenated with x: [real; fake] -> 2B samples.

        Returns:
            DetectorOutput with logits, probabilities, and features
        """
        # Handle paired input
        if pair_x is not None:
            x = torch.cat([x, pair_x], dim=0)

        # Extract features
        features = self._extract_features(x)

        # Temporal aggregation: mean-pool over T when backbone returns [B, T, D]
        if features.dim() == 3:
            features = features.mean(dim=1)

        # Classification
        head_out = self.head(features)
        logits = head_out.output
        prob = F.softmax(logits, dim=1)

        return DetectorOutput(
            logits=logits,
            probabilities=prob,
            features=features,
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone with optional FP32 override.

        Args:
            x: Input tensor [B, C, H, W] or [B, T, C, H, W]

        Returns:
            Feature tensor [B, D]
        """
        if self.backbone_force_fp32:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                backbone_out = self.backbone(x.float())
        else:
            backbone_out = self.backbone(x)

        features = backbone_out.features

        # L2 normalization is handled by the backbone if config.l2_normalize=True.
        # But if backbone doesn't handle it (e.g., timm fallback), apply here.
        if self.l2_normalize and not self.config.backbone.l2_normalize:
            features = F.normalize(features, p=2, dim=-1)

        return features

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self._extract_features(x)

    def get_param_groups(
        self,
        lr: float,
        lr_backbone_mult: float = 0.01,
        weight_decay: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """
        Get parameter groups for optimizer.

        Returns separate groups for backbone (low LR) and head (full LR).
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        groups = []

        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr * lr_backbone_mult,
                "weight_decay": 0.0,  # No decay on backbone (following paper)
                "name": "backbone",
            })

        if head_params:
            groups.append({
                "params": head_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "head",
            })

        return groups

    @property
    def embed_dim(self) -> int:
        """Embedding dimension that the loss function will receive."""
        return self.backbone.get_output_dim()
