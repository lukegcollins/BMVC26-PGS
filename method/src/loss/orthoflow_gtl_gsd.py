"""
OrthoPairGTL + GSD combined loss for Phase II.

Runs the full Paper I loss (CE + OPA alignment + uniformity + GTL) and adds
the three GSD set-geometric regularisers on top.  The CE term is contributed
only by the OPA component to avoid double-counting.

Registered as: "orthoflow_gtl_gsd", "orthoflow_gsd", "opa_gsd"

All GSD hyperparameters live in LossConfig.extra under gsd_* keys (see
gsd_set_loss.py for the full list).  The OPA/GTL hyperparameters are the
same fields used by orthoflow_gtl.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from src.configs import LossConfig
from src.loss.abstract_loss import BaseLoss, LossOutput
from src.loss.gsd_set_loss import GSDSetLoss
from src.loss.orthoflow_gtl import OrthoPairGTLLoss

if TYPE_CHECKING:
    from src.detectors.base_detector import DetectorOutput


class OrthoPairGTLGSDLoss(BaseLoss):
    """
    Combines OrthoPairGTL (Paper I) with GSD set-geometric losses (Phase II).

    Forward pass:
      total = L_opa_total  +  δ_conc * L_conc  +  δ_coh * L_coh  +  δ_disp * L_disp

    where L_opa_total = α*CE + β*align + (γ*GTL) + uniformity  (unchanged from Paper I).
    """

    def __init__(self, config: LossConfig, embed_dim: int = 1024) -> None:
        super().__init__(config)

        # Full Paper I loss
        self.opa = OrthoPairGTLLoss(config, embed_dim=embed_dim)

        # GSD regularisers — CE suppressed to avoid double-counting
        gsd_config = LossConfig(**{**vars(config)})
        gsd_config.extra = {**(config.extra or {}), "gsd_include_ce": False}
        self.gsd = GSDSetLoss(gsd_config)

    def step(self) -> None:
        super().step()
        self.opa.step()

    def set_eval_mode(self, eval_mode: bool = True) -> None:
        super().set_eval_mode(eval_mode)
        self.opa.set_eval_mode(eval_mode)

    def forward(
        self,
        outputs: "DetectorOutput",
        targets: torch.Tensor,
        pair_types: list[str] | None = None,
        group_ids: list[str] | None = None,
        **kwargs: Any,
    ) -> LossOutput:
        # --- Paper I objectives ---
        opa_out = self.opa(outputs, targets, pair_types=pair_types, **kwargs)

        # --- GSD set-geometric objectives (CE suppressed) ---
        gsd_out = self.gsd(outputs, targets, group_ids=group_ids, **kwargs)

        total = opa_out.total + gsd_out.total   # gsd.total has no CE term

        components = {**opa_out.components}
        for k, v in gsd_out.components.items():
            if k != "ce":                        # skip the suppressed CE zero
                components[f"gsd_{k}"] = v

        weights = {**opa_out.weights}
        for k, w in gsd_out.weights.items():
            if k != "ce":
                weights[f"gsd_{k}"] = w

        metrics = {**opa_out.metrics, **{f"gsd_{k}": v for k, v in gsd_out.metrics.items()}}

        return LossOutput(total=total, components=components, weights=weights, metrics=metrics)
