#!/usr/bin/env python3
"""Regenerate main-paper Fig. 2 (fig2_val_curves) from the shipped logs.

Panel (a): macro validation Video-AUC vs epoch, mean +/- SEM over 3 seeds,
           read from the TensorBoard event files under results/logs/main/<run>/tensorboard.
Panel (b): mean Delta Video-AUC vs GenD^DINO over the 17-dataset cross-domain
           suite, bucketed by acquisition modality (RGB-only vs audio-visual),
           read from the per-checkpoint test_results.json files.

Usage:  python src/figures/make_figure2.py [OUTDIR]   (default OUTDIR: figures/)
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO / "results" / "logs" / "main"

CONFIGS = {"LN": "orthoflow_gtl_ln", "LoRA": "orthoflow_gtl_lora",
           "GSD-LoRA": "orthoflow_gtl_gsd_lora"}
SEEDS = ["s42", "s137", "s1729"]
COLORS = {"LN": "#1f77b4", "LoRA": "#ff7f0e", "GSD-LoRA": "#2ca02c"}

# macro-best checkpoint per (config, seed) -- the epoch used in main Table 3.
CKPT = {
    ("LN", "s42"): "epoch18_best_macro", ("LN", "s137"): "epoch24_best_macro",
    ("LN", "s1729"): "epoch09_best_macro",
    ("LoRA", "s42"): "epoch04_best_macro", ("LoRA", "s137"): "epoch05_best_macro",
    ("LoRA", "s1729"): "epoch01_best_macro",
    ("GSD-LoRA", "s42"): "epoch07_best_macro", ("GSD-LoRA", "s137"): "epoch03_best_macro",
    ("GSD-LoRA", "s1729"): "epoch07_best_macro",
}

# test_results.json dataset keys -> paper short names
ALIAS = {"FF-FSH-C23": "FF-FSH", "DFD-C23": "DFD", "DSv1": "DSv1.1",
         "DFDM-C23": "DFDM", "FFIW10K": "FFIW"}
CROSS17 = ["UADFV", "DFD", "CDFv1", "DFDCP", "DFDC", "CDFv2", "FF-FSH", "DeFo",
           "FFIW", "KoDF", "FAVC", "DFDM", "DSv1.1", "PGF", "IDFv1", "DSv2", "CDFv3"]
# Datasets with a synchronised audio track (supp Table S1, "Mod." column).
AV = {"DFDC", "DFDCP", "FAVC", "DSv1.1", "DSv2", "PGF", "IDFv1"}
RGB = [d for d in CROSS17 if d not in AV]
# GenD^DINO re-evaluated on our pipeline (main Table 3).
GEND = dict(zip(CROSS17, [98.88, 97.90, 92.40, 88.82, 84.66, 91.30, 90.02, 98.57,
                          89.31, 90.96, 97.82, 99.81, 87.77, 92.28, 96.66, 80.17, 79.51]))


def val_curves():
    out = {}
    for name, prefix in CONFIGS.items():
        rows = []
        for s in SEEDS:
            ea = EventAccumulator(str(MAIN / f"{prefix}_{s}" / "tensorboard"),
                                  size_guidance={"scalars": 0})
            ea.Reload()
            rows.append([e.value * 100 for e in ea.Scalars("val/video_auc_macro")])
        out[name] = np.array(rows)
    return out


def cross_deltas():
    out = {}
    for name, prefix in CONFIGS.items():
        per_seed = []
        for s in SEEDS:
            p = MAIN / f"{prefix}_{s}" / "test" / CKPT[(name, s)] / "test_results.json"
            ds = json.loads(p.read_text())["datasets"]
            per_seed.append({ALIAS.get(k, k): v["video_metrics"]["auc"] * 100
                             for k, v in ds.items()})
        m = {k: mean(r[k] for r in per_seed) for k in CROSS17}
        d = {k: m[k] - GEND[k] for k in CROSS17}
        out[name] = (mean(d[k] for k in RGB), mean(d[k] for k in AV),
                     mean(d[k] for k in CROSS17))
    return out


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "figures"
    curves, deltas = val_curves(), cross_deltas()
    for k, (r, a, t) in deltas.items():
        print(f"{k:9s} RGB-10={r:+.4f}  AV-7={a:+.4f}  MEAN-17={t:+.4f}")

    plt.rcParams.update({"font.size": 8, "axes.grid": True,
                         "grid.color": "#cccccc", "grid.linewidth": 0.5,
                         "grid.linestyle": "-", "axes.edgecolor": "#333333",
                         "axes.linewidth": 0.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6167, 2.8167))

    # ---- (a) validation trajectory -----------------------------------------
    ep = np.arange(curves["LN"].shape[1])
    for name in CONFIGS:
        y = curves[name]
        mu, sem = y.mean(0), y.std(0, ddof=1) / np.sqrt(y.shape[0])
        ax1.plot(ep, mu, "-o", ms=2.6, lw=1.2, color=COLORS[name], label=name)
        ax1.fill_between(ep, mu - sem, mu + sem, color=COLORS[name], alpha=0.18, lw=0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Macro val Video-AUC (%)")
    ax1.set_title("(a) Validation trajectory", loc="left", fontsize=9)
    ax1.set_ylim(45, 87)
    ax1.set_yticks(range(45, 90, 5))
    ax1.legend(loc="lower right", frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # ---- (b) asymmetry by modality -----------------------------------------
    ax2.grid(axis="y")
    ax2.grid(axis="x", visible=False)
    width, xs = 0.26, np.arange(2)
    for i, name in enumerate(CONFIGS):
        rgb, av, _ = deltas[name]
        vals = [rgb, av]
        pos = xs + (i - 1) * width
        ax2.bar(pos, vals, width * 0.92, color=COLORS[name], alpha=0.45,
                edgecolor=COLORS[name], linewidth=1.0, label=name, zorder=3)
        for x, v in zip(pos, vals):
            ax2.annotate(f"{v:+.2f}", (x, v), textcoords="offset points",
                         xytext=(0, 3 if v >= 0 else -11), ha="center", fontsize=8,
                         zorder=4)
    ax2.axhline(0, color="#333333", lw=0.9, zorder=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["Identity-swap (RGB, 10 sets)", "Reenactment / AV (7 sets)"])
    ax2.set_ylabel(r"Mean $\Delta$ Video-AUC vs GenD$^{\mathrm{DINO}}$ (pp)")
    ax2.set_title("(b) Asymmetry by manipulation modality", loc="left", fontsize=9)
    ax2.set_ylim(-0.42, 2.35)
    ax2.legend(loc="upper left", frameon=False, fontsize=8, ncol=3,
               columnspacing=1.1, handlelength=1.3, handletextpad=0.5)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout(pad=0.6, w_pad=2.0)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig2_val_curves.{ext}", dpi=300)
    print("wrote", outdir / "fig2_val_curves.{pdf,png}")


if __name__ == "__main__":
    main()
