#!/usr/bin/env python3
"""
Generate ablation + sensitivity-sweep configs for the unified-framework
Table 3 rebuild and the rebuttal hyperparameter sweeps.

Each entry in ABLATIONS gets one YAML per seed in {42, 137, 1729}. Each
entry in SWEEPS gets a single seed-42 YAML (single-seed is the rebuttal-
phase strategy under our compute budget). Per-config YAML is produced by
deep-copying a base config (LN or LoRA main run), applying a small delta
dict, and re-serialising. PyYAML round-trips lose the pedagogical comments
in the base configs — acceptable for ablation/sweep files since the deltas
are the only thing that distinguishes them, and the deltas are documented
here in one place.

Run from this directory:
    python3 _generate.py

The resulting YAMLs land in sibling subdirectories: ABLATIONS in
<ablation_name>/, SWEEPS in sweeps/<sweep_name>/.
"""

from __future__ import annotations
import copy
import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # configs/
PARENT = os.path.dirname(HERE)  # orthopair-gtl/

BASE_LN = os.path.join(PARENT, "layer_norm_gtl", "orthoflow_gtl_ln_s42.yaml")
BASE_LORA = os.path.join(PARENT, "lora_gtl", "orthoflow_gtl_lora_s42.yaml")
BASE_GSD_LORA = os.path.join(PARENT, "lora_gtl_gsd", "orthoflow_gtl_gsd_lora_s42.yaml")

SEEDS = (42, 137, 1729)
SWEEP_SEED = 42  # single-seed for the rebuttal-phase sensitivity sweeps


# -----------------------------------------------------------------------------
# Ablation spec: name -> {base, deltas}
#
# `deltas` is a flat dict; keys are dotted paths into the YAML. Lists and
# scalars are supported. `loss_config.extra.<key>` paths are supported.
# `task_target` and `manualSeed` are set per-seed in the generator loop and
# should NOT appear in `deltas`.
# -----------------------------------------------------------------------------
ABLATIONS = {
    # ------------------------------------------------------------------------
    # Frozen probe: backbone fully frozen, linear head, no geometric losses.
    # The bottom-of-table anchor — establishes the floor for what DINOv3
    # features alone can do with a single linear classifier.
    # ------------------------------------------------------------------------
    "frozen_probe": {
        "base": BASE_LN,
        "deltas": {
            "backbone_config.freeze_mode": "full",
            "head_config.type": "linear",
            "loss_config.alpha": 0.0,
            "loss_config.beta": 0.0,
            "loss_config.gamma": 0.0,
            "loss_config.use_gtl": False,
            "loss_config.use_axis_margin": False,
            "loss_config.use_residual_uniformity": False,
            "loss_config.use_paired_slerp_aug": False,
            "loss_config.gtl_small_weight": 0.0,
            "loss_config.extra.gsd_enable_concentration": False,
            "loss_config.extra.gsd_enable_coherence": False,
            "loss_config.extra.gsd_enable_dispersion": False,
        },
    },
    # ------------------------------------------------------------------------
    # LN no-geometric: pure-LayerNorm tuning with CE only.
    # The Row 2 anchor for R2 Q1 ("geometric losses contribute +X over LN
    # alone"). This is the most important ablation row for the rebuttal.
    # ------------------------------------------------------------------------
    "ln_no_geom": {
        "base": BASE_LN,
        "deltas": {
            "loss_config.alpha": 0.0,
            "loss_config.beta": 0.0,
            "loss_config.gamma": 0.0,
            "loss_config.use_gtl": False,
            "loss_config.use_axis_margin": False,
            "loss_config.use_residual_uniformity": False,
            "loss_config.use_paired_slerp_aug": False,
            "loss_config.gtl_small_weight": 0.0,
            "loss_config.extra.gsd_enable_concentration": False,
            "loss_config.extra.gsd_enable_coherence": False,
            "loss_config.extra.gsd_enable_dispersion": False,
        },
    },
    # ------------------------------------------------------------------------
    # LN + OPA only: LayerNorm tuning with the orthogonal-paired-alignment
    # objective enabled but GTL and uniformity disabled. Isolates OPA's
    # contribution on its own.
    # ------------------------------------------------------------------------
    "ln_opa_only": {
        "base": BASE_LN,
        "deltas": {
            "loss_config.alpha": 1.0,
            "loss_config.beta": 0.0,
            "loss_config.gamma": 0.0,
            "loss_config.use_gtl": False,
            "loss_config.use_axis_margin": True,
            "loss_config.use_residual_uniformity": False,
            "loss_config.use_paired_slerp_aug": False,
            "loss_config.gtl_small_weight": 0.0,
            "loss_config.extra.gsd_enable_concentration": False,
            "loss_config.extra.gsd_enable_coherence": False,
            "loss_config.extra.gsd_enable_dispersion": False,
        },
    },
    # ------------------------------------------------------------------------
    # LN + OPA + Uniformity (no GTL): adds the hyperspherical uniformity
    # regulariser (embeddings + residuals) on top of OPA, but no GTL. Isolates the
    # marginal contribution of GTL when stacked on the rest.
    # ------------------------------------------------------------------------
    "ln_opa_unif": {
        "base": BASE_LN,
        "deltas": {
            "loss_config.alpha": 1.0,
            "loss_config.beta": 0.5,
            "loss_config.gamma": 0.0,
            "loss_config.use_gtl": False,
            "loss_config.use_axis_margin": True,
            "loss_config.use_residual_uniformity": True,
            "loss_config.use_paired_slerp_aug": False,
            "loss_config.gtl_small_weight": 0.0,
            "loss_config.extra.gsd_enable_concentration": False,
            "loss_config.extra.gsd_enable_coherence": False,
            "loss_config.extra.gsd_enable_dispersion": False,
        },
    },
    # ------------------------------------------------------------------------
    # LN full method, but with hard real-real negatives turned off. R2 / R1
    # asked specifically about the contribution of the intra-batch hard-
    # negative mining; this row isolates it.
    # ------------------------------------------------------------------------
    "ln_no_hardrr": {
        "base": BASE_LN,
        "deltas": {
            "loss_config.use_hard_rr_negatives": False,
        },
    },
    # ------------------------------------------------------------------------
    # LN full method, but with the GTL small-transform penalty disabled
    # (lambda_small = 0). The old-framework ablation found this drops MEAN-13
    # by ~2.67 points with high seed variance; re-running it under the new
    # framework gives a clean comparison row.
    # ------------------------------------------------------------------------
    "ln_no_small": {
        "base": BASE_LN,
        "deltas": {
            "loss_config.gtl_small_weight": 0.0,
        },
    },
    # ------------------------------------------------------------------------
    # LoRA no-geometric: LoRA backbone with the MLP head but no geometric
    # objectives. The Row 8 anchor — pairs with ln_no_geom to anchor the
    # "geometric losses add value over capacity-only LoRA" comparison.
    # ------------------------------------------------------------------------
    "lora_no_geom": {
        "base": BASE_LORA,
        "deltas": {
            "loss_config.alpha": 0.0,
            "loss_config.beta": 0.0,
            "loss_config.gamma": 0.0,
            "loss_config.use_gtl": False,
            "loss_config.use_axis_margin": False,
            "loss_config.use_residual_uniformity": False,
            "loss_config.use_paired_slerp_aug": False,
            "loss_config.gtl_small_weight": 0.0,
            "loss_config.extra.gsd_enable_concentration": False,
            "loss_config.extra.gsd_enable_coherence": False,
            "loss_config.extra.gsd_enable_dispersion": False,
        },
    },
    # ------------------------------------------------------------------------
    # GSD-LoRA + per-video v̄: replaces the per-pair OPA direction
    # v = (z_f - z_r) / ||z_f - z_r|| with the L2-normalised per-video mean
    # direction v̄. With per-pair v, the residual-consistency term in Eq. 4
    # is identically zero (zr_res ≡ zf_res by construction); v̄ breaks that
    # tautology and lets the residual term carry real gradient. Only
    # meaningful on the GSD-LoRA tier (LN-tier configs do not enable the
    # video-grouped sampler). Run single-seed (s42); s137/s1729 YAMLs are
    # emitted for completeness but were not part of the reported single-seed
    # vbar ablation.
    # ------------------------------------------------------------------------
    "gsd_lora_vbar": {
        "base": BASE_GSD_LORA,
        "deltas": {
            "loss_config.use_per_video_v": True,
        },
    },
    # ------------------------------------------------------------------------
    # LoRA + OPA + Uniformity (no GTL): intermediate point between LoRA-no-
    # geom and full LoRA+OPA+GTL. Mirrors ln_opa_unif on the LoRA capacity
    # tier.
    # ------------------------------------------------------------------------
    "lora_no_gtl": {
        "base": BASE_LORA,
        "deltas": {
            "loss_config.alpha": 1.0,
            "loss_config.beta": 0.5,
            "loss_config.gamma": 0.0,
            "loss_config.use_gtl": False,
            "loss_config.use_axis_margin": True,
            "loss_config.use_residual_uniformity": True,
            "loss_config.use_paired_slerp_aug": False,
            "loss_config.gtl_small_weight": 0.0,
            "loss_config.extra.gsd_enable_concentration": False,
            "loss_config.extra.gsd_enable_coherence": False,
            "loss_config.extra.gsd_enable_dispersion": False,
        },
    },
}


# -----------------------------------------------------------------------------
# Hyperparameter sensitivity sweeps for the rebuttal. All single-seed (s42),
# changing exactly one hyperparameter per config relative to the base full
# method on the corresponding tier (LN-tier for m / lambda_small; GSD-LoRA
# tier for eta_c / eta_h since the GSD terms are only active there).
#
# The default values that already appear in the main runs are listed in
# comments. Sweeps land in sweeps/<name>/orthoflow_gtl_<name>_s42.yaml.
#
# Note on the GTL mixing constant c (R2 Q3): c was previously hardcoded at
# 0.01 in src/loss/_orthoflow_gtl_core.py line 535. Now plumbed through as
# loss_config.gtl_mixing_c (default 0.01 preserved); the sweep_c_* entries
# below exercise it at 0.001 and 0.1.
# -----------------------------------------------------------------------------
SWEEPS = {
    # Sweep the OPA hinge margin m (default in configs: 1.5; R3 Q1)
    "sweep_m_0p5": {"base": BASE_LN, "deltas": {"loss_config.axis_margin": 0.5}},
    "sweep_m_1p0": {"base": BASE_LN, "deltas": {"loss_config.axis_margin": 1.0}},
    "sweep_m_2p0": {"base": BASE_LN, "deltas": {"loss_config.axis_margin": 2.0}},
    "sweep_m_3p0": {"base": BASE_LN, "deltas": {"loss_config.axis_margin": 3.0}},
    # Sweep the GTL small-transform penalty lambda_small (default: 0.2)
    "sweep_lambda_small_0p05": {"base": BASE_LN, "deltas": {"loss_config.gtl_small_weight": 0.05}},
    "sweep_lambda_small_0p5":  {"base": BASE_LN, "deltas": {"loss_config.gtl_small_weight": 0.5}},
    # Sweep the GTL mixing constant c (default: 0.01) — R2 Q3 headline ask.
    # Requires the gtl_mixing_c plumbing committed in src/loss/_orthoflow_gtl_core.py
    # and src/configs/base_config.py.
    "sweep_c_0p001": {"base": BASE_LN, "deltas": {"loss_config.gtl_mixing_c": 0.001}},
    "sweep_c_0p1":   {"base": BASE_LN, "deltas": {"loss_config.gtl_mixing_c": 0.1}},
    # Sweep the GSD weights eta_c = eta_h (default: 5e-3). Uses the GSD-LoRA
    # tier since the GSD terms are only active there.
    "sweep_eta_1e-3": {
        "base": BASE_GSD_LORA,
        "deltas": {
            "loss_config.extra.gsd_delta_concentration": 1e-3,
            "loss_config.extra.gsd_delta_coherence": 1e-3,
        },
    },
    "sweep_eta_2e-2": {
        "base": BASE_GSD_LORA,
        "deltas": {
            "loss_config.extra.gsd_delta_concentration": 2e-2,
            "loss_config.extra.gsd_delta_coherence": 2e-2,
        },
    },
}


def set_deep(cfg: dict, dotted_path: str, value):
    """Set cfg[k1][k2]...=value where dotted_path = 'k1.k2.k3'."""
    parts = dotted_path.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


# -----------------------------------------------------------------------------
# Universal deltas applied to every generated config (after the per-entry
# deltas). Shrinks on-disk checkpoint footprint per run from the main
# runs' ~5 checkpoints (top-3 macro + per-val-dataset bests + last +
# average) down to a single macro-best checkpoint, since the rebuttal
# table reads only the macro-best result. Saves ~6x disk per run with
# no eval impact.
# -----------------------------------------------------------------------------
UNIVERSAL_DELTAS = {
    "checkpoint_config.save_top_k": 1,
    "checkpoint_config.save_last": False,
    "checkpoint_config.save_per_dataset": False,
    "checkpoint_config.save_average": False,
}


# -----------------------------------------------------------------------------
# Extra deltas applied ONLY to sweep configs (after UNIVERSAL_DELTAS).
# Sweeps run at a reduced 10-epoch budget — the main runs peak at epochs
# 3-24 across configurations, so 10 epochs comfortably captures the GSD-LoRA
# tier peaks (3-8) and is on the boundary for the LN tier (9-24). The
# sensitivity sweep is therefore reported as a directional test rather
# than a fully-converged comparison; the rebuttal text should flag this.
# lr_cycle_epochs is lowered alongside nEpochs so the cosine LR schedule
# completes one full cycle within the truncated training run rather than
# leaving the optimiser stuck in the high-LR phase the whole time.
# -----------------------------------------------------------------------------
SWEEP_EXTRA_DELTAS = {
    "nEpochs": 10,
    "lr_cycle_epochs": 10,
}


def _emit(name: str, spec: dict, seed_list: tuple, parent_subdir: str, kind: str) -> int:
    """Write per-seed YAMLs for one ablation or sweep entry; return count written."""
    out_dir = os.path.join(HERE, parent_subdir, name) if parent_subdir != "." else os.path.join(HERE, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(spec["base"]) as f:
        base_cfg = yaml.safe_load(f)
    written = 0
    for seed in seed_list:
        cfg = copy.deepcopy(base_cfg)
        for path, value in spec["deltas"].items():
            set_deep(cfg, path, value)
        for path, value in UNIVERSAL_DELTAS.items():
            set_deep(cfg, path, value)
        if kind == "Sweep":
            for path, value in SWEEP_EXTRA_DELTAS.items():
                set_deep(cfg, path, value)
        cfg["manualSeed"] = seed
        cfg["task_target"] = f"orthoflow_gtl_{name}_s{seed}"
        out = os.path.join(out_dir, f"orthoflow_gtl_{name}_s{seed}.yaml")
        with open(out, "w") as f:
            f.write(
                "# =============================================================================\n"
                f"# {kind}: {name}  (seed={seed})\n"
                "# Auto-generated by ablations/_generate.py — do not edit by hand.\n"
                "# To change behaviour, update the relevant dict in _generate.py and re-run.\n"
                "# =============================================================================\n\n"
            )
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, width=200)
        written += 1
    return written


def main() -> int:
    for path in (BASE_LN, BASE_LORA, BASE_GSD_LORA):
        if not os.path.exists(path):
            print(f"missing base: {path}", file=sys.stderr)
            return 1

    n_abl = 0
    for name, spec in ABLATIONS.items():
        n_abl += _emit(name, spec, SEEDS, parent_subdir=".", kind="Ablation")
    n_swp = 0
    for name, spec in SWEEPS.items():
        n_swp += _emit(name, spec, (SWEEP_SEED,), parent_subdir="sweeps", kind="Sweep")
    print(f"wrote {n_abl} ablation configs + {n_swp} sweep configs under {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
