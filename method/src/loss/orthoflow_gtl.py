"""
OrthoPairGTL Loss — BaseLoss adapter for DeepfakeWorkbench.

This module wraps the battle-tested OrthoPairGTLLoss (from the paper implementation)
to conform to the framework's BaseLoss interface (forward(outputs, targets) -> LossOutput).

The core math lives in the inner OrthoPairGTLCore class (migrated from losses.py).
This adapter handles:
  - Extracting logits/features from DetectorOutput
  - Converting the (total, stats_dict) return format to LossOutput
  - Eval-mode switching (CE-only during validation)

IMPORTANT: The inner loss class is identical to the original losses.py.
Do not modify the core math without re-running the ablation suite.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.configs import LossConfig
from src.loss.abstract_loss import BaseLoss, LossOutput

logger = logging.getLogger(__name__)


class OrthoPairGTLLoss(BaseLoss):
    """
    Framework-conformant wrapper around OrthoPairGTLCore.

    Accepts DetectorOutput and targets, extracts logits + features,
    delegates to the core loss, and returns LossOutput.

    Config mapping (LossConfig -> core kwargs):
        alpha  -> alpha (alignment prior weight)
        beta   -> beta (uniformity prior weight)
        gamma  -> gamma (GTL prior weight)
        use_gtl -> use_gtl
        ... (see LossConfig for full list)
    """

    def __init__(self, config: LossConfig, embed_dim: int = 1024):
        super().__init__(config)
        self._eval_mode = False

        # Import the core loss — either from the existing location or inline
        # This avoids duplicating 700 lines of numerically-sensitive code
        try:
            from src.loss._orthoflow_gtl_core import OrthoPairGTLCore
        except ImportError:
            # Fallback: try the uploaded/original module
            logger.warning(
                "Could not import OrthoPairGTLCore from src.loss._orthoflow_gtl_core. "
                "Ensure the core loss module is placed there."
            )
            raise

        self.core = OrthoPairGTLCore(
            embed_dim=embed_dim,
            weighting_mode=config.weighting_mode,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            debug_log_params=config.debug_log_params,
            use_gtl=config.use_gtl,
            gtl_disc_weight=config.gtl_disc_weight,
            gtl_small_weight=config.gtl_small_weight,
            gtl_hidden_dim=config.gtl_hidden_dim,
            gtl_output_dim=config.gtl_output_dim,
            use_hard_rr_negatives=config.use_hard_rr_negatives,
            residual_loss=config.residual_loss,
            use_axis_margin=config.use_axis_margin,
            axis_margin=config.axis_margin,
            uniformity_t=config.uniformity_t,
            use_residual_uniformity=config.use_residual_uniformity,
            residual_uniformity_t=config.residual_uniformity_t,
            use_paired_slerp_aug=config.use_paired_slerp_aug,
            slerp_pair_multiplier=config.slerp_pair_multiplier,
            ce_only_in_eval=config.ce_only_in_eval,
        )

        self.embed_dim = embed_dim
        logger.info(f"OrthoPairGTLLoss adapter initialized (embed_dim={embed_dim})")

    def set_eval_mode(self, eval_mode: bool = True) -> None:
        """Toggle eval mode (CE-only during validation)."""
        self._eval_mode = eval_mode

    def forward(
        self,
        outputs: Any,
        targets: torch.Tensor,
        pair_types: list[str] | None = None,
        **kwargs,
    ) -> LossOutput:
        """
        Compute OrthoPairGTL loss.

        Args:
            outputs: DetectorOutput with .logits [B, C] and .features [B, D]
            targets: Ground truth labels [B]
            pair_types: Optional pair type annotations (unused, kept for API compat)

        Returns:
            LossOutput with total loss, components dict, and metrics dict
        """
        # Extract from DetectorOutput
        logits = outputs.logits
        features = outputs.features

        if features is None:
            raise ValueError(
                "OrthoPairGTLLoss requires features (embeddings) from DetectorOutput. "
                "Ensure the detector returns features in its output."
            )

        # Sync training mode with the adapter's eval flag
        was_training = self.core.training
        if self._eval_mode:
            self.core.eval()
        else:
            self.core.train(was_training)

        # Delegate to core loss
        total_loss, stats = self.core(logits, features, targets)

        # Restore training state
        self.core.train(was_training)

        # Build LossOutput
        components = {}
        metrics = {}

        # Map stats dict to components (tensors) and metrics (floats)
        for key, value in stats.items():
            if key == "total":
                continue  # Skip, we have total_loss already
            if isinstance(value, (int, float)):
                if key in ("ce", "align", "uni", "gtl", "gtl_disc", "gtl_small"):
                    # These are loss component values — create tensors for logging
                    components[key] = torch.tensor(value, device=total_loss.device)
                else:
                    metrics[key] = float(value)
            elif isinstance(value, torch.Tensor):
                components[key] = value

        # Extract weights for logging
        weights = {}
        for key in ("w_align", "w_uni", "w_gtl"):
            if key in stats:
                weights[key.replace("w_", "")] = float(stats[key])

        return LossOutput(
            total=total_loss,
            components=components,
            weights=weights,
            metrics=metrics,
        )
