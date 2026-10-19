"""
Set-Geometric Descriptor (GSD) losses for OrthoPairGTL Phase II.

Implements three set-level geometric regularisers over the OPA decomposition
(v_i, ψ_i, z⊥_i) derived from real-fake embedding pairs:

  L_concentration : Fake-video displacement directions should concentrate
                    (high C_dir = ||mean(v_i)||).
  L_coherence     : Residual subspaces z⊥_i should be stable across frames
                    for every video (high C_resid = mean pairwise cos-sim).
  L_dispersion    : Mean-direction vectors across videos/groups should spread
                    out on the hypersphere (uniformity loss), preventing all
                    fake signatures collapsing to the same GSD fingerprint.
                    (NOTE: Disabled by default until multi-step accumulation
                    or very large batches are used).

CRITICAL DATALOADER CONTRACT
----------------------------
This loss requires strict pairing. The batch must contain corresponding real
and fake frames in the exact same relative order.
For example, an interleaved batch `[real_1, fake_1, real_2, fake_2...]` or
a split batch `[real_1...real_N, fake_1...fake_N]` where real_i is the direct
source frame for fake_i. Shuffled batches without explicit pair matching will
destroy the geometric manifold.

Operation modes
---------------
video-level (group_ids provided)
    Frames are grouped by video ID.  Per-group statistics drive all three
    losses.  L_concentration requires ≥ min_group_size frames per group.
    Pass group_ids as a flat list aligned with the batch order.

batch-level (no group_ids)
    All N real-fake pairs in the batch are treated as one virtual set.
    L_concentration is *skipped* in this mode — batch-wide C_dir conflates
    within-video coherence with cross-video method diversity and is not a
    meaningful signal for mixed batches.  L_coherence and L_dispersion are
    still applied.

Config keys (all in LossConfig.extra)
--------------------------------------
gsd_delta_concentration  float  0.05   weight for L_concentration
gsd_delta_coherence      float  0.05   weight for L_coherence
gsd_delta_dispersion     float  0.02   weight for L_dispersion
gsd_real_c_dir_threshold float  0.30   (Reserved for future real penalty)
gsd_min_group_size       int    2      minimum frames to form a valid group
gsd_uniformity_t         float  2.0    temperature for Wang & Isola kernel
gsd_enable_concentration bool   True   toggle L_concentration
gsd_enable_coherence     bool   True   toggle L_coherence
gsd_enable_dispersion    bool   False  toggle L_dispersion (needs large N)
gsd_include_ce           bool   True   include CE in total
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.configs import LossConfig

# Re-use numerically stable primitives from the core OPA module.
from src.loss._orthoflow_gtl_core import (  # noqa: PLC2701
    _EPS,
    _safe_normalize,
    _uniformity_loss,
)
from src.loss.abstract_loss import BaseLoss, LossOutput

if TYPE_CHECKING:
    from src.detectors.base_detector import DetectorOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Differentiable GSD primitives
# ---------------------------------------------------------------------------


def _opa_decompose(
    zr: torch.Tensor,
    zf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-frame OPA decomposition for a set of N real-fake pairs.

    Args:
        zr: [N, D] real embeddings (unit-normalised on hypersphere)
        zf: [N, D] fake embeddings (unit-normalised on hypersphere)

    Returns:
        v     : [N, D] unit manipulation directions  (Δ / ‖Δ‖)
        psi   : [N]    geodesic angles  arccos(zr · zf)  [radians]
        z_perp: [N, D] residuals of zf orthogonal to v
    """
    # Unit manipulation direction
    delta = zf - zr  # [N, D]
    v = _safe_normalize(delta, dim=1)  # [N, D]

    # Geodesic distance (angle on unit hypersphere)
    cos_psi = (zr * zf).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    psi = torch.acos(cos_psi)  # [N]

    # Residual component of zf orthogonal to v (unnormalised)
    proj = (zf * v).sum(dim=1, keepdim=True)  # [N, 1]
    z_perp = zf - proj * v  # [N, D]

    return v, psi, z_perp


def _c_dir(v: torch.Tensor) -> torch.Tensor:
    """
    Directional concentration C_dir = ‖mean(v_i)‖ ∈ [0, 1].

    High C_dir means all displacement vectors point in roughly the
    same direction (characteristic of identity-swap methods).
    """
    return torch.norm(v.float().mean(dim=0), p=2)  # scalar


def _c_resid(z_perp: torch.Tensor) -> torch.Tensor:
    """
    Residual coherence C_resid = mean off-diagonal cosine similarity
    of the normalised residual vectors.
    """
    N = z_perp.shape[0]
    if N < 2:
        return z_perp.new_zeros([])

    # Normalise residuals before computing similarity
    z_n = _safe_normalize(z_perp, dim=1).float()  # [N, D]
    sim = z_n @ z_n.t()  # [N, N]

    # Extract strictly upper triangular indices to save memory
    i, j = torch.triu_indices(N, N, offset=1, device=z_perp.device)
    return sim[i, j].mean()  # scalar


# ---------------------------------------------------------------------------
# Group parsing helpers
# ---------------------------------------------------------------------------


def _build_group_index(
    group_ids: list[str],
    labels: torch.Tensor,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """
    Split group_ids into real-sample groups and fake-sample groups.
    """
    real_groups: dict[str, list[int]] = {}
    fake_groups: dict[str, list[int]] = {}

    for i, (gid, lbl) in enumerate(zip(group_ids, labels.tolist())):
        if lbl == 0:
            real_groups.setdefault(gid, []).append(i)
        else:
            fake_groups.setdefault(gid, []).append(i)

    return real_groups, fake_groups


# ---------------------------------------------------------------------------
# Main loss class
# ---------------------------------------------------------------------------


class GSDSetLoss(BaseLoss):
    """
    Set-Geometric Descriptor losses for Phase II OrthoPairGTL.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__(config)
        extra: dict = config.extra or {}

        self.delta_concentration: float = float(extra.get("gsd_delta_concentration", 0.05))
        self.delta_coherence: float = float(extra.get("gsd_delta_coherence", 0.05))
        self.delta_dispersion: float = float(extra.get("gsd_delta_dispersion", 0.02))
        self.min_group_size: int = int(extra.get("gsd_min_group_size", 2))
        self.uniformity_t: float = float(extra.get("gsd_uniformity_t", 2.0))

        self.enable_concentration: bool = bool(extra.get("gsd_enable_concentration", True))
        self.enable_coherence: bool = bool(extra.get("gsd_enable_coherence", True))
        # Disabled by default due to high dimensionality vs small group sizes
        self.enable_dispersion: bool = bool(extra.get("gsd_enable_dispersion", False))
        self.include_ce: bool = bool(extra.get("gsd_include_ce", True))

        self._ce = nn.CrossEntropyLoss()

    def _group_losses(
        self,
        zr: torch.Tensor,  # [N, D]
        zf: torch.Tensor,  # [N, D]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (l_conc, l_coh, v_mean) for one paired group of N items.
        """
        zero = zr.new_zeros([])
        N = zr.shape[0]

        v, _psi, z_perp = _opa_decompose(zr, zf)

        # -- L_concentration (Fakes only) --
        if self.enable_concentration:
            cd = _c_dir(v)
            l_conc = -torch.log(cd.clamp_min(_EPS))
        else:
            l_conc = zero

        # -- L_coherence --
        if self.enable_coherence and N >= 2:
            l_coh = -_c_resid(z_perp)  # maximise coherence
        else:
            l_coh = zero

        v_mean = v.float().mean(dim=0)  # [D]
        return l_conc, l_coh, v_mean

    def forward(
        self,
        outputs: DetectorOutput,
        targets: torch.Tensor,
        group_ids: list[str] | None = None,
        **kwargs,
    ) -> LossOutput:
        # --- CE ---
        l_ce = self._ce(outputs.logits, targets.long()) if self.include_ce else outputs.logits.new_zeros([])

        features = outputs.features  # [B, D]
        B = features.shape[0]

        # Extract strict pairs.
        # WARNING: This assumes `targets` ordering guarantees real_indices[i]
        # corresponds perfectly to fake_indices[i].
        real_indices = (targets == 0).nonzero(as_tuple=True)[0]
        fake_indices = (targets == 1).nonzero(as_tuple=True)[0]
        n_pairs = min(len(real_indices), len(fake_indices))

        if n_pairs < 1:
            return LossOutput(
                total=l_ce,
                components={"ce": l_ce},
                weights={"ce": 1.0},
                metrics={"n_pairs": 0.0, "n_groups": 0.0},
            )

        # Trim to matched pairs
        zr_pool = features[real_indices[:n_pairs]]
        zf_pool = features[fake_indices[:n_pairs]]

        l_conc_terms: list[torch.Tensor] = []
        l_coh_terms: list[torch.Tensor] = []
        v_means: list[torch.Tensor] = []

        if group_ids is not None and len(group_ids) == B:
            # ---- Video-level grouping ----
            real_groups, fake_groups = _build_group_index(group_ids, targets)

            # Map global index -> paired pool index
            real_pos = {int(idx.item()): pos for pos, idx in enumerate(real_indices[:n_pairs])}
            fake_pos = {int(idx.item()): pos for pos, idx in enumerate(fake_indices[:n_pairs])}

            for gid, f_idxs in fake_groups.items():
                r_idxs = real_groups.get(gid, [])

                # Collect positions within the trimmed pool tensors
                f_positions = [fake_pos[i] for i in f_idxs if i in fake_pos]
                r_positions = [real_pos[i] for i in r_idxs if i in real_pos]

                n_g = min(len(f_positions), len(r_positions) if r_positions else len(f_positions))
                if n_g < self.min_group_size:
                    continue

                zf_g = zf_pool[f_positions[:n_g]]
                zr_g = zr_pool[r_positions[:n_g]] if r_positions else zr_pool[:n_g]

                lc, lh, vm = self._group_losses(zr_g, zf_g)
                l_conc_terms.append(lc)
                l_coh_terms.append(lh)
                v_means.append(vm)

            # NOTE: Real-group penalty loop removed. We only concentrate fake vectors.

        else:
            # ---- Batch-level fallback ----
            _lc_unused, lh, vm = self._group_losses(zr_pool, zf_pool)
            l_coh_terms.append(lh)
            v_means.append(vm)

            if group_ids is not None and len(group_ids) != B:
                logger.warning(
                    "GSDSetLoss: group_ids length %d ≠ batch size %d — falling back to batch-level mode.",
                    len(group_ids),
                    B,
                )

        zero = features.new_zeros([])

        l_conc = torch.stack(l_conc_terms).mean() if l_conc_terms else zero
        l_coh = torch.stack(l_coh_terms).mean() if l_coh_terms else zero

        # L_dispersion: Wang & Isola uniformity over mean direction vectors
        if self.enable_dispersion and len(v_means) >= 2:
            v_stack = torch.stack(v_means, dim=0)  # [G, D]
            l_disp = _uniformity_loss(v_stack, t=self.uniformity_t)
        else:
            l_disp = zero

        total = l_ce + self.delta_concentration * l_conc + self.delta_coherence * l_coh + self.delta_dispersion * l_disp

        return LossOutput(
            total=total,
            components={
                "ce": l_ce,
                "concentration": l_conc,
                "coherence": l_coh,
                "dispersion": l_disp,
            },
            weights={
                "ce": 1.0,
                "concentration": self.delta_concentration,
                "coherence": self.delta_coherence,
                "dispersion": self.delta_dispersion,
            },
            metrics={
                "n_pairs": float(n_pairs),
                "n_groups": float(len(v_means)),
            },
        )
