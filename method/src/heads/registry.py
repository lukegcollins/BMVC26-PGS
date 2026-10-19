"""
Head registry for DeepfakeWorkbench.

Provides factory functions for creating head instances from configuration.
Maps head type strings from YAML configs to concrete Head classes.
"""

import logging

from src.configs import HeadConfig
from src.core.registry import StrategyRegistry
from src.heads.base import BaseHead

logger = logging.getLogger(__name__)

# Global head registry
head_registry = StrategyRegistry("head", BaseHead)


def build_head(config: HeadConfig, input_dim: int) -> BaseHead:
    """
    Build a head from configuration.

    Args:
        config: HeadConfig instance (config.type selects the head class)
        input_dim: Input feature dimension from backbone

    Returns:
        Instantiated head

    Raises:
        ValueError: If head type not found in registry

    Example:
        config = HeadConfig(type="mlp", hidden_dim=512, num_classes=2, dropout=0.1)
        head = build_head(config, input_dim=1024)
    """
    head_type = config.type.lower()

    if head_type not in head_registry:
        available = ", ".join(sorted(head_registry.list_all_names()))
        raise ValueError(f"Unknown head type: '{head_type}'. Available: [{available}]")

    head = head_registry.create(head_type, config=config, input_dim=input_dim)
    return head


def register_head(cls: type[BaseHead], *names: str) -> None:
    """
    Register a head class with the global registry.

    Args:
        cls: Head class (must inherit from BaseHead)
        *names: Names/aliases to register under
    """
    head_registry.register_class(cls, *names)


# -------------------------------------------------------------------------
# Register built-in heads
# -------------------------------------------------------------------------
def _register_builtin_heads():
    """Register built-in head implementations."""
    from src.heads.classification import (
        LinearHead,
        MLP3LayerHead,
        MLPBatchNormHead,
        MLPHead,
    )

    register_head(LinearHead, "linear", "linear_head")
    register_head(MLPHead, "mlp", "mlp_head")
    register_head(MLP3LayerHead, "mlp_3layer", "mlp3", "mlp_3_layer")
    register_head(MLPBatchNormHead, "mlp_bn", "mlp_batchnorm", "mlp_batch_norm")

    # Optional: temporal heads (if they conform to BaseHead)
    try:
        from src.detectors.temporal_heads import (
            TemporalConv1DHead,
            TemporalGRUHead,
            TemporalLSTMHead,
            TemporalMLPHead,
            TemporalTransformerHead,
        )

        # These need an adapter since they have different __init__ signatures.
        # For now, register only if they inherit from BaseHead.
        # TODO: Create proper adapters or refactor temporal heads to use BaseHead.
        logger.debug("Temporal heads available but not yet registered in head_registry (pending adapter)")
    except ImportError:
        pass


_register_builtin_heads()
