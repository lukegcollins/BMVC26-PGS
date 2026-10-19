"""
DINOv3 backbone implementation (production-ready).

Key guarantees:
- No silent input corruption when Albumentations Normalize() is used.
- Explicit control over input format via config.extra:
    - input_already_normalized: bool  (default False)
    - input_in_0_255: bool | None     (default None -> inferred safely)
- Robust loading priority:
    1) local checkpoint (if provided)
    2) HuggingFace (preferred)
    3) torch.hub
    4) timm fallback (optional, default disabled)
- LoRA support via PEFT (freeze_mode="lora")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.configs import BackboneConfig
from src.backbones.base import BackboneOutput, BaseBackbone

logger = logging.getLogger(__name__)

# HF optional import
try:
    from transformers import AutoModel

    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# PEFT optional import
try:
    from peft import LoraConfig, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# -------------------------------------------------------------------------
# Model configs
# -------------------------------------------------------------------------

DINOV3_CONFIGS: dict[str, dict[str, int]] = {
    "dinov3_vits16": {"embed_dim": 384, "patch_size": 16, "num_heads": 6, "depth": 12},
    "dinov3_vitb16": {"embed_dim": 768, "patch_size": 16, "num_heads": 12, "depth": 12},
    "dinov3_vitl16": {"embed_dim": 1024, "patch_size": 16, "num_heads": 16, "depth": 24},
    "dinov3_vitg16": {"embed_dim": 1536, "patch_size": 16, "num_heads": 24, "depth": 40},
    "dinov3_vith16": {"embed_dim": 1280, "patch_size": 16, "num_heads": 16, "depth": 32},
    "dinov3_vitl14": {"embed_dim": 1024, "patch_size": 14, "num_heads": 16, "depth": 24},
    "dinov3_vitg14": {"embed_dim": 1536, "patch_size": 14, "num_heads": 24, "depth": 40},
}

HF_MODEL_MAP = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vitg16": "facebook/dinov3-vitg16-pretrain-lvd1689m",
    "dinov3_vith16": "facebook/dinov3-vith16-pretrain-lvd1689m",
    "dinov3_vitl14": "facebook/dinov3-vitl14-pretrain-lvd1689m",
    "dinov3_vitg14": "facebook/dinov3-vitg14-pretrain-lvd1689m",
}


class DINOv3Backbone(BaseBackbone):
    """
    DINOv3 Vision Transformer backbone.

    Freeze modes:
        - "none": All parameters trainable
        - "full": All parameters frozen
        - "layernorm": Freeze all except LayerNorm affine params (~0.03%)
        - "partial": Freeze first N layers (config.extra.freeze_layers)
        - "lora": Freeze base model, inject LoRA adapters via PEFT (~8.5%)

    LoRA config (via config.extra):
        - lora_r: int (default 16)
        - lora_alpha: int (default 32)
        - lora_dropout: float (default 0.05)
        - lora_target_modules: list[str] (default ["q_proj", "v_proj"])
    """

    def __init__(self, config: BackboneConfig):
        super().__init__(config)

        self.model_name = config.name
        self.model_config = DINOV3_CONFIGS.get(config.name, self._infer_config(config.name))

        self._hf_model = False
        self.model = self._load_model(config)

        self._embed_dim = int(self.model_config["embed_dim"])
        self.patch_size = int(self.model_config["patch_size"])
        self.depth = int(self.model_config["depth"])

        # Apply freezing strategy (including LoRA if requested)
        if config.freeze_mode == "lora":
            self._apply_lora()
        else:
            self._apply_freeze()

        self._log_params()

        logger.info(
            f"Initialized DINOv3 backbone: {config.name}, "
            f"embed_dim={self._embed_dim}, patch={self.patch_size}, depth={self.depth}, "
            f"freeze_mode={config.freeze_mode}"
        )

    # ---------------------------------------------------------------------
    # LoRA / PEFT
    # ---------------------------------------------------------------------

    def _apply_lora(self) -> None:
        """
        Apply LoRA adaptation to the backbone via PEFT.

        Reads LoRA hyperparameters from config.lora dict (preferred):
            - rank: Adapter rank (default 16)
            - alpha: Scaling factor (default 32)
            - dropout: Dropout on LoRA layers (default 0.05)
            - target_modules: Which modules to target (default ["q_proj", "v_proj"])
        Falls back to legacy config.extra flat keys: lora_r, lora_alpha, lora_dropout, lora_target_modules.
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "freeze_mode='lora' requires the 'peft' library. "
                "Install with: pip install peft"
            )

        # Read from structured lora: dict (rank/alpha/dropout/target_modules),
        # falling back to legacy extra flat keys (lora_r/lora_alpha/etc.).
        lora_cfg = self.config.lora or {}
        extra = self.config.extra or {}
        r = int(lora_cfg.get("rank", extra.get("lora_r", 16)))
        alpha = int(lora_cfg.get("alpha", extra.get("lora_alpha", 32)))
        dropout = float(lora_cfg.get("dropout", extra.get("lora_dropout", 0.05)))
        targets = lora_cfg.get("target_modules", extra.get("lora_target_modules", ["q_proj", "v_proj"]))

        logger.info(f"Applying LoRA to DINOv3 backbone: r={r}, alpha={alpha}, targets={targets}")

        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            bias="none",
            task_type=None,
            inference_mode=False,
        )

        # Wrap model with PEFT
        self.model = get_peft_model(self.model, peft_config)

        # Log parameter counts
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pct = 100.0 * trainable / max(1, total)
        logger.info(f"LoRA injected: {trainable:,} / {total:,} trainable ({pct:.2f}%)")

        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

    def _log_params(self) -> None:
        """Log trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pct = 100.0 * trainable / max(1, total)
        logger.info(f"Backbone params: {trainable:,} / {total:,} trainable ({pct:.3f}%)")

    # ---------------------------------------------------------------------
    # Config / Loading (unchanged from original)
    # ---------------------------------------------------------------------

    def _infer_config(self, name: str) -> dict[str, int]:
        cfg = {"embed_dim": 1024, "patch_size": 16, "num_heads": 16, "depth": 24}
        n = name.lower()
        if "vits" in n:
            cfg.update({"embed_dim": 384, "num_heads": 6, "depth": 12})
        elif "vitb" in n:
            cfg.update({"embed_dim": 768, "num_heads": 12, "depth": 12})
        elif "vitl" in n:
            cfg.update({"embed_dim": 1024, "num_heads": 16, "depth": 24})
        elif "vitg" in n:
            cfg.update({"embed_dim": 1536, "num_heads": 24, "depth": 40})
        elif "vith" in n:
            cfg.update({"embed_dim": 1280, "num_heads": 16, "depth": 32})

        if "14" in n:
            cfg["patch_size"] = 14
        elif "16" in n:
            cfg["patch_size"] = 16
        return cfg

    def _load_model(self, config: BackboneConfig) -> nn.Module:
        extra = config.extra or {}
        allow_timm_fallback = bool(extra.get("allow_timm_fallback", False))

        # 1) Local checkpoint
        if config.weights_dir:
            weights_path = Path(config.weights_dir)
            if weights_path.is_file():
                ckpt = weights_path
            else:
                ckpt = weights_path / f"{config.name}.pth"
                if not ckpt.exists():
                    ckpt = weights_path / f"{config.name}_pretrain.pth"

            if ckpt.exists():
                logger.info(f"Loading DINOv3 from local checkpoint: {ckpt}")
                return self._load_from_checkpoint(ckpt)

        # 2) HuggingFace
        if HF_AVAILABLE:
            m = self._load_from_huggingface(config)
            if m is not None:
                return m

        # 3) torch.hub
        hub_repo = extra.get("hub_repo", "facebookresearch/dinov3")
        try:
            logger.info(f"Attempting to load DINOv3 from torch.hub: repo={hub_repo}, name={config.name}")
            model = torch.hub.load(hub_repo, config.name, pretrained=True)
            return model
        except Exception as e:
            logger.warning(f"torch.hub load failed: {e}")

        # 4) timm fallback (optional, default off)
        if not allow_timm_fallback:
            raise RuntimeError(
                "Failed to load DINOv3 from local weights, HuggingFace, or torch.hub, "
                "and allow_timm_fallback=False (good). Fix loading rather than training a generic ViT."
            )

        logger.warning("Falling back to timm ViT (NOT DINOv3 weights).")
        return self._create_vit_model(config)

    def _load_from_huggingface(self, config: BackboneConfig) -> nn.Module | None:
        extra = config.extra or {}
        hf_id = extra.get("hf_id", HF_MODEL_MAP.get(config.name, config.name))

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if config.weights_dir:
            load_kwargs["cache_dir"] = config.weights_dir

        try:
            logger.info(f"Loading DINOv3 from HuggingFace: {hf_id}")
            model = AutoModel.from_pretrained(hf_id, **load_kwargs)
            self._hf_model = True
            return model
        except Exception as e:
            logger.warning(f"Could not load DINOv3 from HuggingFace ({hf_id}): {e}")
            self._hf_model = False
            return None

    def _load_from_checkpoint(self, checkpoint_path: Path) -> nn.Module:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "teacher" in checkpoint:
            state_dict = checkpoint["teacher"]
        else:
            state_dict = checkpoint

        state_dict = {k.replace("backbone.", "").replace("module.", ""): v for k, v in state_dict.items()}

        model = self._create_vit_model(self.config)
        model.load_state_dict(state_dict, strict=False)
        return model

    def _create_vit_model(self, config: BackboneConfig) -> nn.Module:
        import timm

        embed_dim = int(self.model_config["embed_dim"])
        patch = int(self.model_config["patch_size"])
        if embed_dim == 384:
            size = "small"
        elif embed_dim == 768:
            size = "base"
        elif embed_dim == 1024:
            size = "large"
        elif embed_dim == 1536:
            size = "giant"
        else:
            size = "large"

        timm_name = f"vit_{size}_patch{patch}_224"
        logger.info(f"Creating ViT via timm: {timm_name}")
        return timm.create_model(timm_name, pretrained=True, num_classes=0)

    # ---------------------------------------------------------------------
    # Input handling
    # ---------------------------------------------------------------------

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for ViT.

        If Albumentations Normalize() is already applied, you MUST set:
            backbone_config.extra.input_already_normalized: true
        """
        extra = self.config.extra or {}
        already_norm = bool(extra.get("input_already_normalized", False))

        if already_norm:
            return x

        input_in_0_255 = extra.get("input_in_0_255", None)

        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            if input_in_0_255 is True:
                x = x.div(255.0)
            elif input_in_0_255 is None:
                if x.max().item() > 1.5:
                    x = x.div(255.0)

        x = self.normalize_input(x)
        return x

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        is_sequence = x.dim() == 5
        if is_sequence:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        else:
            B = x.shape[0]
            T = 1

        x = self._prepare_input(x)

        with torch.amp.autocast("cuda", enabled=not self.config.force_fp32):
            cls_token, patch_tokens = self._extract_features(x)

        if self.config.l2_normalize and cls_token is not None:
            cls_token = nn.functional.normalize(cls_token, p=2, dim=-1)

        if is_sequence:
            cls_token = cls_token.view(B, T, -1)
            if patch_tokens is not None:
                patch_tokens = patch_tokens.view(B, T, patch_tokens.shape[1], -1)

        return BackboneOutput(
            features=cls_token,
            cls_token=cls_token,
            patch_tokens=patch_tokens,
        )

    def _extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # HuggingFace (including PEFT-wrapped HF models)
        if getattr(self, "_hf_model", False):
            # PEFT wraps the HF model but preserves the forward signature.
            # pixel_values goes through the base model transparently.
            outputs = self.model(pixel_values=x)

            # Handle both direct HF output and PEFT-wrapped output
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                hidden = outputs[0]
            if hidden is None:
                raise RuntimeError("HF DINOv3 model returned no last_hidden_state.")

            cls_token = hidden[:, 0]
            patch_tokens = hidden[:, 1:]
            del outputs, hidden
            return cls_token, patch_tokens

        # torch.hub / timm
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(x)
            if isinstance(out, dict):
                cls_token = out.get("x_norm_clstoken", out.get("cls_token"))
                patch_tokens = out.get("x_norm_patchtokens", out.get("patch_tokens"))
                return cls_token, patch_tokens
            if hasattr(out, "dim") and out.dim() == 3:
                return out[:, 0], out[:, 1:]
            return out, None

        # Fallback
        out = self.model(x)
        if hasattr(out, "dim") and out.dim() == 3:
            return out[:, 0], out[:, 1:]
        return out, None

    def get_output_dim(self) -> int:
        return self._embed_dim
