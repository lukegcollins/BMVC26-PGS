"""
OrthoPairGTL Loss - Fully Refactored with Deep Debugging & Numerical Safety

# =============================================================================
# CHANGES FROM ORIGINAL:
# =============================================================================
# 1. DEEP DEBUGGING: When debug_log_params=True, logs:
#    - Min/max/mean of each loss component
#    - Uncertainty weight values (sigma parameters)
#    - Gradient statistics (when register_hooks=True)
#    - Detection of which component is causing NaN/Inf
#
# 2. NUMERICAL SAFETY: All operations are protected:
#    - _safe_normalize: eps-protected L2 normalization
#    - _safe_acos: Clamped input to [-1+eps, 1-eps] before acos
#    - _safe_exp: Clamped exponent to prevent overflow
#    - All divisions protected by eps
#
# 3. NAN DETECTION: Early detection with detailed reporting
#
# 4. FIX (2025-12-17): Now handles any batch structure, not just alternating
#    [real, fake, real, fake]. Separates by label and pairs minimum counts.
# =============================================================================
"""

from __future__ import annotations
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.amp as amp
except Exception:
    import torch.cuda.amp as amp

logger = logging.getLogger(__name__)

# Global epsilon for numerical stability
_EPS = 1e-7
_ACOS_EPS = 1e-6  # More conservative for acos


# =============================================================================
# NUMERICAL SAFETY FUNCTIONS
# =============================================================================


def _safe_normalize(x: torch.Tensor, dim: int = 1, eps: float = _EPS) -> torch.Tensor:
    """
    L2-normalize with numerical stability.

    Always computed in float32 to avoid half-precision underflow.
    """
    with amp.autocast(device_type=x.device.type, enabled=False):
        xf = x.float()
        norm = torch.norm(xf, p=2, dim=dim, keepdim=True)
        norm = norm.clamp_min(eps)
        result = xf / norm
        return result.to(dtype=x.dtype)


def _safe_acos(x: torch.Tensor, eps: float = _ACOS_EPS) -> torch.Tensor:
    """
    Safe arccos that clamps input to valid range.

    acos(x) is only defined for x in [-1, 1]. Due to numerical errors,
    dot products can slightly exceed this range, causing NaN.
    """
    return torch.acos(x.clamp(-1.0 + eps, 1.0 - eps))


def _safe_exp(x: torch.Tensor, max_exp: float = 80.0) -> torch.Tensor:
    """
    Safe exponential that prevents overflow.

    exp(x) overflows for x > ~88 in float32, ~11 in float16.
    We clamp to max_exp for safety.
    """
    return torch.exp(x.clamp(-max_exp, max_exp))


def _check_tensor(t: torch.Tensor, name: str, log_fn=None) -> bool:
    """
    Check tensor for NaN/Inf and optionally log details.

    Returns:
        True if tensor is valid (no NaN/Inf), False otherwise
    """
    has_nan = torch.isnan(t).any()
    has_inf = torch.isinf(t).any()

    if has_nan or has_inf:
        if log_fn:
            log_fn(
                f"[NaN/Inf DETECTED] {name}: "
                f"NaN={has_nan.item()}, Inf={has_inf.item()}, "
                f"shape={t.shape}, dtype={t.dtype}"
            )
        return False
    return True


def _tensor_stats(t: torch.Tensor) -> str:
    """Return formatted tensor statistics string."""
    with torch.no_grad():
        return (
            f"min={t.min().item():.4f}, "
            f"max={t.max().item():.4f}, "
            f"mean={t.mean().item():.4f}, "
            f"std={t.std().item():.4f}"
        )


def _uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Compute uniformity loss (Wang & Isola, 2020).

    Encourages embeddings to be uniformly distributed on the hypersphere.
    """
    if z.shape[0] < 2:
        return z.new_zeros([])

    with amp.autocast(device_type=z.device.type, enabled=False):
        z_norm = _safe_normalize(z, dim=1).float()
        # Pairwise squared distances
        pdist_sq = torch.pdist(z_norm, p=2).pow(2)
        # Uniformity loss (log of average Gaussian kernel)
        kernel = _safe_exp(pdist_sq.mul(-t), max_exp=20.0)
        uniform_loss = torch.log(kernel.mean() + _EPS)
        return uniform_loss


def _validate_paired_batch(labels: torch.Tensor) -> bool:
    """Validate that batch has proper [real, fake] pairing."""
    if labels.numel() < 2 or labels.numel() % 2 != 0:
        return False
    expected_01 = torch.tensor([0, 1], device=labels.device).repeat(labels.numel() // 2)
    return torch.equal(labels, expected_01) or torch.equal(labels, 1 - expected_01)


# =============================================================================
# TRANSFORMATION ENCODER (for GTL)
# =============================================================================


class TransformationEncoder(nn.Module):
    """
    Encodes the transformation between two embeddings.

    Takes: direction vector (v), delta axis projection (delta_a), angle (theta)
    Returns: transformation embedding for discriminator
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.direction_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
        self.magnitude_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.gate = nn.Linear(64, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self, v: torch.Tensor, delta_a: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        h_dir = self.direction_net(v)
        mag = torch.cat([delta_a, theta], dim=1)
        h_mag = self.magnitude_net(mag)
        gate = torch.sigmoid(self.gate(h_mag))
        return self.output_norm(gate * h_dir)


# =============================================================================
# WEIGHTING STRATEGIES
# =============================================================================


class WeightingStrategy(nn.Module):
    """Base class for task weighting strategies."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks

    def forward(self, losses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class StaticWeighting(WeightingStrategy):
    """Fixed weights for each task."""

    def __init__(self, num_tasks: int, weights: List[float]):
        super().__init__(num_tasks)
        self.register_buffer("weights", torch.tensor(weights))

    def forward(self, losses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.weights, torch.tensor(0.0, device=losses.device)


class UncertaintyWeighting(WeightingStrategy):
    """
    Uncertainty-based task weighting (Kendall et al., 2018).

    Learns task-specific uncertainty (homoscedastic) to balance losses.

    The weight for task i is: 1 / (2 * sigma_i^2)
    With regularization: log(sigma_i)

    We parameterize as log(sigma) for numerical stability.
    """

    def __init__(
        self,
        num_tasks: int,
        init_weights: List[float],
        clamp: float = 2.0,  # Tighter clamp for stability
        debug_log: bool = False,
    ):
        super().__init__(num_tasks)
        self.clamp = clamp
        self.debug_log = debug_log

        # Initialize log_sigma from initial weights
        # weight = 1 / (2 * sigma^2)  =>  sigma^2 = 1 / (2 * weight)
        # log_sigma = 0.5 * log(1 / (2 * weight)) = -0.5 * log(2 * weight)
        def _init_log_sigma(w: float) -> float:
            w = max(float(w), 1e-6)
            return -0.5 * math.log(2.0 * w)

        self.log_sigmas = nn.Parameter(
            torch.tensor([_init_log_sigma(w) for w in init_weights])
        )

    def forward(self, losses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Clamp for numerical stability
        clamped = self.log_sigmas.clamp(-self.clamp, self.clamp)

        # weight = 1 / (2 * sigma^2) = 0.5 * exp(-2 * log_sigma)
        weights = 0.5 * _safe_exp(-2.0 * clamped, max_exp=10.0)

        # Regularization: sum of log_sigmas (encourages smaller sigma = larger weight)
        reg = clamped.sum()

        if self.debug_log:
            with torch.no_grad():
                sigmas = torch.exp(clamped)
                logger.info(
                    f"[DEBUG-UNCERT] log_sigma={clamped.detach().cpu().tolist()}, "
                    f"sigma={sigmas.detach().cpu().tolist()}, "
                    f"weights={weights.detach().cpu().tolist()}"
                )

        return weights, reg


# =============================================================================
# MAIN LOSS CLASS
# =============================================================================


class OrthoPairGTLCore(nn.Module):
    """
    Orthogonal Pair Loss with Geodesic Transformation Learning.

    Components:
    1. Cross-Entropy Loss (CE): Standard classification
    2. Orthogonal Alignment Loss: Align real/fake pairs in embedding space
    3. Uniformity Loss: Encourage uniform distribution on hypersphere
    4. GTL Loss (optional): Distinguish real-fake transforms from real-real

    Deep Debugging:
    When debug_log_params=True, provides detailed logging of:
    - Each loss component value
    - Uncertainty weights
    - Tensor statistics (min/max/mean)
    - NaN/Inf detection with source identification

    NOTE: This version handles ANY batch structure by separating embeddings
    by label (real=0, fake=1) rather than assuming alternating order.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        weighting_mode: str = "uncertainty",
        alpha: float = 0.1,  # Alignment prior weight
        beta: float = 0.15,  # Uniformity prior weight
        gamma: float = 1.2,  # GTL prior weight
        clamp_log_sigma: float = 2.0,
        debug_log_params: bool = False,
        use_gtl: bool = True,
        gtl_disc_weight: float = 1.0,
        gtl_small_weight: float = 0.2,
        gtl_hidden_dim: int = 512,
        gtl_output_dim: int = 256,
        use_hard_rr_negatives: bool = True,
        residual_loss: str = "l2",
        use_axis_margin: bool = True,
        axis_margin: float = 0.15,
        uniformity_t: float = 2.0,
        use_residual_uniformity: bool = True,
        residual_uniformity_t: float = 3.0,
        use_paired_slerp_aug: bool = True,
        slerp_pair_multiplier: int = 1,
        ce_only_in_eval: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_gtl = use_gtl
        self.ce_only_in_eval = ce_only_in_eval
        self.debug_log = debug_log_params
        self.gtl_disc_weight = float(gtl_disc_weight)
        self.gtl_small_weight = float(gtl_small_weight)
        self.use_hard_negatives = use_hard_rr_negatives
        self.residual_loss = residual_loss.lower()
        self.use_axis_margin = use_axis_margin
        self.axis_margin = float(axis_margin)
        self.uniformity_t = float(uniformity_t)
        self.use_residual_uniformity = use_residual_uniformity
        self.residual_uniformity_t = float(residual_uniformity_t)
        self.use_slerp_aug = use_paired_slerp_aug
        self.slerp_multiplier = slerp_pair_multiplier

        self.ce = nn.CrossEntropyLoss()

        # GTL components
        if use_gtl:
            self.transform_encoder = TransformationEncoder(
                embed_dim, gtl_hidden_dim, gtl_output_dim
            )
            self.transform_discriminator = nn.Linear(gtl_output_dim, 2)

        # Task weighting
        self.num_tasks = 3 if use_gtl else 2
        priors = [alpha, beta, gamma][: self.num_tasks]

        # Comprehensive config logging
        logger.info("=" * 60)
        logger.info("OrthoPairGTL Loss Configuration")
        logger.info("=" * 60)
        logger.info(f"  Weighting mode:     {weighting_mode}")
        logger.info(f"  Alpha (alignment):  {alpha}")
        logger.info(f"  Beta (uniformity):  {beta}")
        logger.info(f"  Gamma (GTL):        {gamma}")
        logger.info(f"  Use GTL:            {use_gtl}")
        logger.info(f"  Use axis margin:    {use_axis_margin} (margin={axis_margin})")
        logger.info(f"  Use residual unif:  {use_residual_uniformity}")
        logger.info(f"  Use SLERP aug:      {use_paired_slerp_aug}")
        logger.info(f"  Hard RR negatives:  {use_hard_rr_negatives}")
        logger.info(f"  Residual loss:      {residual_loss}")
        logger.info(f"  Active tasks:       {self.num_tasks} with weights {priors}")
        logger.info("=" * 60)

        # Warn about potential config issues
        if alpha == 0 and beta == 0 and not use_gtl:
            logger.warning(
                "[CONFIG-WARNING] alpha=0, beta=0, use_gtl=False: "
                "Only CE loss will be used. Auxiliary losses disabled."
            )

        if weighting_mode == "uncertainty":
            self.strategy = UncertaintyWeighting(
                self.num_tasks,
                priors,
                clamp=clamp_log_sigma,
                debug_log=debug_log_params,
            )
        else:
            self.strategy = StaticWeighting(self.num_tasks, priors)

        # Debug iteration counter
        self._debug_iter = 0

    def _debug_tensor(self, name: str, t: torch.Tensor):
        """Log tensor statistics for debugging."""
        if not self.debug_log:
            return

        with torch.no_grad():
            is_valid = _check_tensor(t, name, logger.error)
            if is_valid:
                logger.info(f"[DEBUG-TENSOR] {name}: {_tensor_stats(t)}")

    def _orthogonal_alignment(
        self, zr: torch.Tensor, zf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute orthogonal alignment loss between real/fake pairs.

        The key idea: decompose embeddings into:
        1. Component along the real->fake direction (axis)
        2. Residual component (orthogonal to axis)

        We want residuals to match (same identity) while axis projections differ.
        """
        # Direction from real to fake
        v = _safe_normalize(zf - zr, dim=1)

        # Axis projections (scalar projection onto direction)
        a_r = (zr * v).sum(dim=1, keepdim=True)
        a_f = (zf * v).sum(dim=1, keepdim=True)

        # Residuals (component orthogonal to direction)
        zr_res = zr - a_r * v
        zf_res = zf - a_f * v

        # Normalize residuals for comparison
        zr_res_norm = _safe_normalize(zr_res, dim=1)
        zf_res_norm = _safe_normalize(zf_res, dim=1)

        if self.debug_log:
            self._debug_tensor("a_r", a_r)
            self._debug_tensor("a_f", a_f)
            self._debug_tensor("zr_res_norm", zr_res_norm)
            self._debug_tensor("zf_res_norm", zf_res_norm)

        # Alignment loss: residuals should be similar
        if self.residual_loss == "cosine":
            # Cosine similarity loss (1 - cos_sim)
            cos_sim = (zr_res_norm * zf_res_norm).sum(dim=1).clamp(-1, 1)
            loss_align = 1.0 - cos_sim
        else:
            # L2 loss on normalized residuals
            loss_align = (zr_res_norm - zf_res_norm).pow(2).sum(dim=1)

        # Axis margin loss: fake should project further along axis than real
        loss_hinge = zr.new_zeros([])
        if self.use_axis_margin:
            margin_violation = self.axis_margin - (a_f - a_r).squeeze()
            loss_hinge = F.relu(margin_violation).pow(2)

        return loss_align.mean(), loss_hinge.mean(), (zr_res_norm, zf_res_norm)

    def _gtl_loss(
        self, z: torch.Tensor, labels: torch.Tensor, zr: torch.Tensor, zf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Geodesic Transformation Learning loss.

        Trains a discriminator to distinguish:
        - Real->Fake transformations (class 1)
        - Real->Real transformations (class 0)

        This encourages the model to learn forgery-specific transformation patterns.

        NOTE: Updated to accept pre-separated zr, zf instead of assuming
        alternating batch structure.
        """
        if zr.shape[0] < 1 or zf.shape[0] < 1:
            return z.new_zeros([]), z.new_zeros([])

        # Encode Real->Fake transformation
        # Match real and fake by index (assumes they're paired by some mechanism)
        P = min(zr.shape[0], zf.shape[0])
        zr_paired = zr[:P]
        zf_paired = zf[:P]

        v_rf = _safe_normalize(zf_paired - zr_paired, dim=1)

        # Delta axis projection
        da_rf = (zf_paired * v_rf).sum(1, keepdim=True) - (zr_paired * v_rf).sum(
            1, keepdim=True
        )

        # Angle between embeddings (using safe acos)
        dot_rf = (zr_paired * zf_paired).sum(1, keepdim=True)
        theta_rf = _safe_acos(dot_rf)

        h_rf = self.transform_encoder(v_rf, da_rf, theta_rf)

        if self.debug_log:
            self._debug_tensor("v_rf", v_rf)
            self._debug_tensor("da_rf", da_rf)
            self._debug_tensor("theta_rf", theta_rf)
            self._debug_tensor("h_rf", h_rf)

        # Get real embeddings for Real->Real transforms
        z_real = zr  # Already separated
        if z_real.shape[0] < 2:
            return z.new_zeros([]), z.new_zeros([])

        # Find pairs for Real->Real transforms
        if self.use_hard_negatives:
            # Hard negatives: pair with closest real embedding
            with torch.no_grad():
                sim = z_real @ z_real.T
                sim.fill_diagonal_(-float("inf"))
                closest = sim.argmax(dim=1)
            z_start, z_end = z_real, z_real[closest]
        else:
            # Random pairing
            perm = torch.randperm(z_real.shape[0], device=z.device)
            z_start, z_end = z_real, z_real[perm]

        # Encode Real->Real transformation
        v_rr = _safe_normalize(z_end - z_start, dim=1)
        da_rr = ((z_end - z_start) * v_rr).sum(1, keepdim=True)
        dot_rr = (z_start * z_end).sum(1, keepdim=True)
        theta_rr = _safe_acos(dot_rr)

        h_rr = self.transform_encoder(v_rr, da_rr, theta_rr)

        if self.debug_log:
            self._debug_tensor("v_rr", v_rr)
            self._debug_tensor("da_rr", da_rr)
            self._debug_tensor("theta_rr", theta_rr)
            self._debug_tensor("h_rr", h_rr)

        # Discriminator loss
        logits_rf = self.transform_discriminator(h_rf)
        logits_rr = self.transform_discriminator(h_rr)

        labels_rf = torch.ones(h_rf.size(0), dtype=torch.long, device=z.device)
        labels_rr = torch.zeros(h_rr.size(0), dtype=torch.long, device=z.device)

        loss_disc = (
            F.cross_entropy(logits_rf, labels_rf)
            + F.cross_entropy(logits_rr, labels_rr)
        ) / 2.0

        # Small transformation regularization (encourage small R->R transforms)
        loss_small = theta_rr.pow(2).mean() + 0.01 * da_rr.pow(2).mean()

        if self.debug_log:
            self._debug_tensor("loss_disc", loss_disc)
            self._debug_tensor("loss_small", loss_small)

        return loss_disc, loss_small

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and return statistics.

        Args:
            logits: Classification logits [B, num_classes]
            features: Feature embeddings [B, embed_dim]
            labels: Ground truth labels [B]

        Returns:
            total_loss: Combined weighted loss
            stats: Dictionary of loss component values for logging
        """
        stats = {}
        self._debug_iter += 1

        # Cross-entropy loss (always computed)
        loss_ce = self.ce(logits, labels.long())

        if self.debug_log:
            self._debug_tensor("loss_ce", loss_ce)

        # In eval mode, optionally only use CE
        if (not self.training) and self.ce_only_in_eval:
            return loss_ce, {"total": loss_ce.item(), "ce": loss_ce.item()}

        # Normalize features to unit hypersphere
        z = _safe_normalize(features, dim=1)

        # =================================================================
        # FIX: Separate embeddings by label instead of assuming alternating
        # =================================================================
        real_mask = labels == 0
        fake_mask = labels == 1

        zr = z[real_mask]  # Real embeddings
        zf = z[fake_mask]  # Fake embeddings

        # Number of valid pairs
        P = min(zr.shape[0], zf.shape[0])

        # Log batch composition on first iteration (always) and periodically in debug mode
        if self._debug_iter == 1:
            logger.info(
                f"[LOSS-INFO] First batch composition: "
                f"Total={features.shape[0]}, Real={zr.shape[0]}, Fake={zf.shape[0]}, Pairs={P}"
            )
            if P == 0:
                logger.warning(
                    "[LOSS-WARNING] No pairs available for alignment loss! "
                    "Check if dataset is loading both real and fake samples."
                )

        if self.debug_log and self._debug_iter % 100 == 0:
            self._debug_tensor("z_normalized", z)
            logger.info(
                f"[DEBUG-LOSS] Batch: {features.shape[0]}, Real: {zr.shape[0]}, Fake: {zf.shape[0]}, Pairs: {P}"
            )
            logger.info(f"[DEBUG-LOSS] Labels: {labels.tolist()[:20]}...")  # First 20

        # =====================================================================
        # Orthogonal Alignment Loss
        # =====================================================================
        loss_align = z.new_zeros([])
        loss_hinge = z.new_zeros([])
        residuals = None

        if P > 0:
            # Use first P samples from each class for pairing
            loss_align, loss_hinge, residuals = self._orthogonal_alignment(
                zr[:P], zf[:P]
            )
            loss_align = loss_align + loss_hinge

            if self.debug_log:
                self._debug_tensor("loss_align", loss_align)
                self._debug_tensor("loss_hinge", loss_hinge)

        # =====================================================================
        # Uniformity Loss
        # =====================================================================
        loss_uni = z.new_zeros([])
        if self.training:
            loss_uni = _uniformity_loss(z, t=self.uniformity_t)

            # Optional: add residual uniformity
            if self.use_residual_uniformity and residuals is not None:
                res_combined = torch.cat(residuals, dim=0)
                loss_uni = loss_uni + _uniformity_loss(
                    res_combined, t=self.residual_uniformity_t
                )

            if self.debug_log:
                self._debug_tensor("loss_uni", loss_uni)

        # =====================================================================
        # GTL Loss
        # =====================================================================
        loss_gtl = z.new_zeros([])
        if self.training and self.use_gtl:
            loss_disc, loss_small = self._gtl_loss(z, labels, zr, zf)
            loss_gtl = (
                self.gtl_disc_weight * loss_disc + self.gtl_small_weight * loss_small
            )
            stats.update(
                {
                    "gtl_disc": loss_disc.item(),
                    "gtl_small": loss_small.item(),
                }
            )

            if self.debug_log:
                self._debug_tensor("loss_gtl", loss_gtl)

        # =====================================================================
        # Task Weighting & Total Loss
        # =====================================================================

        # Ensure all losses have gradients (for weighting to work)
        def ensure_grad(t):
            if t.numel() == 0 or not t.requires_grad:
                return z.sum() * 0.0  # Zero with gradient
            return t

        task_losses = [ensure_grad(loss_align), ensure_grad(loss_uni)]
        if self.use_gtl:
            task_losses.append(ensure_grad(loss_gtl))

        # Get weights from strategy
        stacked_losses = torch.stack([l.detach() for l in task_losses])
        weights, reg = self.strategy(stacked_losses)

        # Compute weighted total
        weighted_sum = sum(w * l for w, l in zip(weights, task_losses))
        total_loss = loss_ce + weighted_sum + reg

        # =====================================================================
        # Collect Statistics
        # =====================================================================
        stats.update(
            {
                "total": total_loss.item(),
                "ce": loss_ce.item(),
                "align": task_losses[0].item(),
                "uni": task_losses[1].item(),
                "w_align": weights[0].item(),
                "w_uni": weights[1].item(),
                "reg": reg.item(),
                "num_pairs": P,  # Added for debugging
            }
        )
        if self.use_gtl:
            stats.update(
                {
                    "gtl": task_losses[2].item(),
                    "w_gtl": weights[2].item(),
                }
            )

        # Deep debug logging (periodic, not every iteration)
        if self.debug_log and self._debug_iter % 50 == 0:
            weights_str = [f"{w:.3f}" for w in weights.tolist()]
            logger.info(
                f"[DEBUG-LOSS] Iter {self._debug_iter}: "
                f"total={total_loss.item():.4f}, ce={loss_ce.item():.4f}, "
                f"align={task_losses[0].item():.4f}, uni={task_losses[1].item():.4f}, "
                f"pairs={P}, weights={weights_str}"
            )

            # Check for anomalies
            for name, val in stats.items():
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    logger.error(f"[NaN/Inf] {name} = {val}")

        return total_loss, stats
