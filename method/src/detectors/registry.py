"""
Detector registry for DeepfakeWorkbench.

Provides factory functions for creating complete detector models.
"""

from src.configs import ExperimentConfig
from src.core.registry import StrategyRegistry
from src.detectors.base_detector import BaseDetector

# Global detector registry
detector_registry = StrategyRegistry("detector", BaseDetector)


def build_detector(config: ExperimentConfig) -> BaseDetector:
    """
    Build a detector from configuration.

    Args:
        config: ExperimentConfig instance

    Returns:
        Instantiated detector

    Raises:
        ValueError: If detector model_name not found in registry
    """
    model_name = config.model_name.lower()

    # Try exact match first
    if model_name in detector_registry:
        return detector_registry.create(model_name, config)

    # Try base name (e.g., "orthoflow_gtl" -> "orthoflow")
    base_name = model_name.split("_")[0]
    if base_name in detector_registry:
        return detector_registry.create(base_name, config)

    # Fallback to standard detector
    if "standard" in detector_registry:
        return detector_registry.create("standard", config)

    raise ValueError(f"Unknown detector model: '{model_name}'. Available: {detector_registry.list_strategies()}")


def register_detector(cls: type[BaseDetector], *names: str) -> None:
    """
    Register a detector class with the global registry.

    Args:
        cls: Detector class (must inherit from BaseDetector)
        *names: Names/aliases to register under
    """
    detector_registry.register_class(cls, *names)


# Import implementations to trigger registration
def _register_builtin_detectors():
    """Register built-in detector implementations."""
    # Core OrthoPairGTL detector (always available after migration)
    from src.detectors.orthoflow_detector import OrthoFlowDetector

    register_detector(OrthoFlowDetector, "orthoflow", "orthoflow_gtl", "ortho")

    # Optional: other detector implementations
    try:
        from src.detectors.standard_detector import StandardDetector
        register_detector(StandardDetector, "standard", "base", "default")
    except ImportError:
        pass

    try:
        from src.detectors.orthoflow_temporal_detector import OrthoFlowTemporalDetector
        register_detector(OrthoFlowTemporalDetector, "orthoflow_gtl_temporal", "orthoflow_temporal")
    except ImportError:
        pass

    try:
        from src.detectors.orthoflow_gated_temporal_detector import OrthoFlowGatedTemporalDetector
        register_detector(OrthoFlowGatedTemporalDetector, "orthoflow_gated_temporal", "gated_temporal")
    except ImportError:
        pass

    try:
        from src.detectors.temporal_detector import TemporalDetector, TemporalPairedDetector
        register_detector(TemporalDetector, "temporal", "temporal_detector", "video")
        register_detector(TemporalPairedDetector, "temporal_paired", "paired_temporal", "sequence_paired")
    except ImportError:
        pass


_register_builtin_detectors()
