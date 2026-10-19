"""
DINOv2 backbone implementation.

DINOv2 is a self-supervised Vision Transformer trained by Meta AI.
Reference: https://github.com/facebookresearch/dinov2
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.configs import BackboneConfig
from src.backbones.base import BackboneOutput, BaseBackbone

logger = logging.getLogger(__name__)

# DINOv2 model configurations
DINOV2_CONFIGS = {
    "dinov2_vits14": {"embed_dim": 384, "patch_size": 14, "num_heads": 6, "depth": 12},
    "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14, "num_heads": 12, "depth": 12},
    "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14, "num_heads": 16, "depth": 24},
    "dinov2_vitg14": {"embed_dim": 1536, "patch_size": 14, "num_heads": 24, "depth": 40},
    # With registers
    "dinov2_vits14_reg": {"embed_dim": 384, "patch_size": 14, "num_heads": 6, "depth": 12, "num_register_tokens": 4},
    "dinov2_vitb14_reg": {"embed_dim": 768, "patch_size": 14, "num_heads": 12, "depth": 12, "num_register_tokens": 4},
    "dinov2_vitl14_reg": {"embed_dim": 1024, "patch_size": 14, "num_heads": 16, "depth": 24, "num_register_tokens": 4},
    "dinov2_vitg14_reg": {"embed_dim": 1536, "patch_size": 14, "num_heads": 24, "depth": 40, "num_register_tokens": 4},
}


class DINOv2Backbone(BaseBackbone):
    """
    DINOv2 Vision Transformer backbone.

    Supports all DINOv2 variants (ViT-S/B/L/G with 14x14 patches)
    and versions with register tokens.

    Features:
    - Automatic weight loading from torch.hub or local directory
    - Configurable freezing (full, layernorm, partial)
    - L2 normalization option for contrastive learning
    - Access to intermediate features and patch tokens

    Example:
        config = BackboneConfig(
            name="dinov2_vitl14",
            weights_dir="./pretrained/dinov2",
            freeze_mode="layernorm",
            l2_normalize=True
        )
        backbone = DINOv2Backbone(config)
        output = backbone(images)  # [B, 3, 224, 224] -> BackboneOutput
    """

    def __init__(self, config: BackboneConfig):
        super().__init__(config)

        self.model_name = config.name
        self.model_config = DINOV2_CONFIGS.get(config.name, DINOV2_CONFIGS["dinov2_vitl14"])

        # Load model
        self.model = self._load_model(config)

        # Store dimensions
        self._embed_dim = self.model_config["embed_dim"]
        self.patch_size = self.model_config["patch_size"]

        # Apply freezing strategy
        self._apply_freeze()

        logger.info(
            f"Initialized DINOv2 backbone: {config.name}, embed_dim={self._embed_dim}, freeze_mode={config.freeze_mode}"
        )

    def _load_model(self, config: BackboneConfig) -> nn.Module:
        """Load DINOv2 model from hub or local weights."""
        weights_dir = config.weights_dir

        if weights_dir and Path(weights_dir).exists():
            # Load from local directory
            weights_path = Path(weights_dir) / f"{config.name}.pth"
            if weights_path.exists():
                logger.info(f"Loading DINOv2 from local weights: {weights_path}")
                model = torch.hub.load("facebookresearch/dinov2", config.name, pretrained=False)
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                return model

        # Load from torch hub with pretrained weights
        logger.info(f"Loading DINOv2 from torch.hub: {config.name}")
        model = torch.hub.load("facebookresearch/dinov2", config.name, pretrained=True)
        return model

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """
        Extract features from images.

        Args:
            x: Input images [B, C, H, W] with values in [0, 1] or normalized

        Returns:
            BackboneOutput with cls_token features and patch_tokens
        """
        # Handle sequence input [B, T, C, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            is_sequence = True
        else:
            is_sequence = False
            B = x.shape[0]

        # Normalize if not already done
        if x.max() > 1.0:
            x = x / 255.0
        x = self.normalize_input(x)

        # Forward through DINOv2
        # DINOv2 returns dict with various outputs
        with torch.amp.autocast("cuda", enabled=not self.config.force_fp32):
            if hasattr(self.model, "forward_features"):
                output = self.model.forward_features(x)
                if isinstance(output, dict):
                    cls_token = output.get("x_norm_clstoken", output.get("cls_token"))
                    patch_tokens = output.get("x_norm_patchtokens", output.get("patch_tokens"))
                else:
                    # Some versions return tensor directly
                    cls_token = output[:, 0]
                    patch_tokens = output[:, 1:]
            else:
                # Fallback: use standard forward
                features = self.model(x)
                cls_token = features
                patch_tokens = None

        # L2 normalize if requested
        if self.config.l2_normalize and cls_token is not None:
            cls_token = nn.functional.normalize(cls_token, p=2, dim=-1)

        # Reshape for sequence
        if is_sequence:
            cls_token = cls_token.view(B, T, -1)
            if patch_tokens is not None:
                patch_tokens = patch_tokens.view(B, T, patch_tokens.shape[1], -1)

        return BackboneOutput(
            features=cls_token,
            cls_token=cls_token,
            patch_tokens=patch_tokens,
        )

    def get_output_dim(self) -> int:
        """Return embedding dimension."""
        return self._embed_dim

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        layer_indices: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Get features from intermediate layers.

        Args:
            x: Input images [B, C, H, W]
            layer_indices: Which layers to extract (default: last 4)

        Returns:
            Dict mapping layer names to feature tensors
        """
        if layer_indices is None:
            depth = self.model_config["depth"]
            layer_indices = list(range(depth - 4, depth))

        x = self.normalize_input(x)

        features = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                features[name] = output

            return hook

        # Register hooks on specified layers
        for idx in layer_indices:
            layer = self.model.blocks[idx]
            hook = layer.register_forward_hook(make_hook(f"layer_{idx}"))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            self.model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features
