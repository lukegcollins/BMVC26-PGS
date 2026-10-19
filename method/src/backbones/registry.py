"""
Backbone registry for DeepfakeWorkbench.

Provides factory functions for creating backbone instances from configuration.
"""

from src.configs import BackboneConfig
from src.core.registry import StrategyRegistry
from src.backbones.base import BaseBackbone

# Global backbone registry
backbone_registry = StrategyRegistry("backbone", BaseBackbone)


def build_backbone(config: BackboneConfig) -> BaseBackbone:
    """
    Build a backbone from configuration.

    Args:
        config: BackboneConfig instance

    Returns:
        Instantiated backbone

    Raises:
        ValueError: If backbone name not found in registry

    Example:
        config = BackboneConfig(name="dinov2_vitl14", freeze_mode="layernorm")
        backbone = build_backbone(config)
    """
    # Extract the backbone family from name (e.g., "dinov2_vitl14" -> "dinov2")
    name_parts = config.name.split("_")
    backbone_family = name_parts[0].lower()

    return backbone_registry.create(backbone_family, config)


def register_backbone(cls: type[BaseBackbone], *names: str) -> None:
    """
    Register a backbone class with the global registry.

    Args:
        cls: Backbone class (must inherit from BaseBackbone)
        *names: Names/aliases to register under
    """
    backbone_registry.register_class(cls, *names)


# Import implementations to trigger registration
# This is done at the bottom to avoid circular imports
def _register_builtin_backbones():
    """Register built-in backbone implementations."""
    from src.backbones.dinov2_backbone import DINOv2Backbone
    from src.backbones.dinov3_backbone import DINOv3Backbone

    register_backbone(DINOv2Backbone, "dinov2", "dino_v2", "dino2")
    register_backbone(DINOv3Backbone, "dinov3", "dino_v3", "dino3")

    # Optional: timm backbone
    try:
        from src.backbones.timm_backbone import TimmBackbone
        register_backbone(TimmBackbone, "timm", "efficientnet", "resnet", "convnext", "swin")
    except ImportError:
        pass  # timm backbone not implemented yet

    # Optional: X-CLIP for video understanding
    try:
        from src.backbones.xclip_backbone import XCLIPBackbone

        register_backbone(XCLIPBackbone, "xclip", "x_clip", "xclip_base", "xclip_large")
    except ImportError:
        pass  # X-CLIP dependencies not available

    # Optional: VideoPrism for temporal video understanding
    try:
        from src.backbones.videoprism_backbone import VideoPrismBackbone

        register_backbone(VideoPrismBackbone, "videoprism", "video_prism", "videoprism_base", "videoprism_large")
    except ImportError:
        pass  # VideoPrism dependencies not available

    # Optional: V-JEPA 2 for temporal video understanding
    try:
        from src.backbones.vjepa2_backbone import VJEPA2Backbone

        register_backbone(VJEPA2Backbone, "vjepa2", "v_jepa2", "vjepa2_vitl", "vjepa2_vitg", "vjepa2_vith")
    except ImportError:
        pass  # V-JEPA 2 dependencies not available


_register_builtin_backbones()
