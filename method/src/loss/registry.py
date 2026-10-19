"""
Loss registry for DeepfakeWorkbench.

Provides factory functions for creating loss instances from configuration.
"""

from src.configs import LossConfig
from src.core.registry import StrategyRegistry
from src.loss.abstract_loss import BaseLoss

# Global loss registry
loss_registry = StrategyRegistry("loss", BaseLoss)


def build_loss(config: LossConfig, model_name: str = "standard") -> BaseLoss:
    """
    Build a loss function from configuration.

    Args:
        config: LossConfig instance
        model_name: Model name to help select appropriate loss

    Returns:
        Instantiated loss function

    Raises:
        ValueError: If loss type not found in registry

    Example:
        config = LossConfig(alpha=1.0, beta=0.5, gamma=1.0, use_gtl=True)
        loss_fn = build_loss(config, model_name="orthoflow")
    """
    # Determine loss type based on configuration and model
    if "orthoflow" in model_name.lower():
        loss_type = "orthoflow"
    elif config.use_gtl or config.beta > 0:
        loss_type = "combined"
    else:
        loss_type = "ce"

    return loss_registry.create(loss_type, config)


def register_loss(cls: type[BaseLoss], *names: str) -> None:
    """
    Register a loss class with the global registry.

    Args:
        cls: Loss class (must inherit from BaseLoss)
        *names: Names/aliases to register under
    """
    loss_registry.register_class(cls, *names)


# Import implementations to trigger registration
def _register_builtin_losses():
    """Register built-in loss implementations."""
    # Core OrthoPairGTL loss (always available after migration)
    from src.loss.orthoflow_gtl import OrthoPairGTLLoss

    register_loss(OrthoPairGTLLoss, "ortho_pair_gtl", "orthoflow_gtl", "orthopairgtl", "orthoflow", "orthoflow_loss")

    # Optional: other loss implementations (may not exist yet in migrated codebase)
    try:
        from src.loss.classification import BinaryCrossEntropyLoss, CrossEntropyLoss
        register_loss(CrossEntropyLoss, "ce", "cross_entropy", "crossentropy")
        register_loss(BinaryCrossEntropyLoss, "bce", "binary_cross_entropy")
    except ImportError:
        pass

    try:
        from src.loss.combined import CombinedLoss, OrthoFlowLoss
        register_loss(CombinedLoss, "combined", "multi_task")
    except ImportError:
        pass

    try:
        from src.loss.contrastive import GTLLoss, OrthoPairLoss
        register_loss(OrthoPairLoss, "ortho_pair", "orthopair", "ortho")
        register_loss(GTLLoss, "gtl", "texture")
    except ImportError:
        pass

    try:
        from src.loss.kot_loss import KOTLoss
        register_loss(KOTLoss, "kot", "kinematic_ortho", "kot_loss")
    except ImportError:
        pass

    try:
        from src.loss.pairwise_contrastive import PairwiseContrastiveLoss
        register_loss(PairwiseContrastiveLoss, "pairwise_contrastive", "pairwise", "paired_contrastive")
    except ImportError:
        pass

    try:
        from src.loss.videoprism_losses import SupConLoss, TemporalDerivativeLoss, VideoPrismCombinedLoss
        register_loss(SupConLoss, "supcon", "supervised_contrastive")
        register_loss(TemporalDerivativeLoss, "temporal_deriv", "temporal_derivative")
        register_loss(VideoPrismCombinedLoss, "videoprism", "videoprism_combined")
    except ImportError:
        pass

    try:
        from src.loss.gsd_set_loss import GSDSetLoss
        register_loss(GSDSetLoss, "gsd_set", "gsd", "set_geometric")
    except ImportError:
        pass

    try:
        from src.loss.orthoflow_gtl_gsd import OrthoPairGTLGSDLoss
        register_loss(OrthoPairGTLGSDLoss, "orthoflow_gtl_gsd", "orthoflow_gsd", "opa_gsd")
    except ImportError:
        pass


_register_builtin_losses()
