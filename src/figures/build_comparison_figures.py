#!/usr/bin/env python3
"""Generate publication-ready figures from `comparison_master.csv`:

  Fig. S1  figS1_heatmap.{pdf,png}               -- method × dataset heatmap (video AUC)
  Fig. S2  figS2_per_dataset.{pdf,png}           -- grouped bars per cross-domain dataset
  Fig. S3  figS3_summary.{pdf,png}               -- bar chart of MEAN-13 / MEAN-17 / MEAN-IND
  Fig. S4  figS4_delta_gsd_vs_gend.{pdf,png}     -- per-dataset delta of GSD-LoRA vs GenD-DINOv3

Outputs go to `figures/`.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT_PLOT = ROOT / "figures"
OUT_PLOT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "results/baselines/comparison_master.csv", index_col=0)

# Method display order — baselines (paper order, append FakeSTormer) then ours
ROW_ORDER = ["ForensicAdapter_v1","Effort-AIGI-Detection_FFpp",
             "GenD_CLIP_L_14","GenD_PE_L","GenD_DINOv3_L","FakeSTormer_c23",
             "ln","lora","gsd_lora"]
ROW_LABELS = {
    "ForensicAdapter_v1": "ForAda",
    "Effort-AIGI-Detection_FFpp": "Effort",
    "GenD_CLIP_L_14": "GenD$^{\\mathrm{CLIP}}$",
    "GenD_PE_L":      "GenD$^{\\mathrm{PE}}$",
    "GenD_DINOv3_L":  "GenD$^{\\mathrm{DINO}}$",
    "FakeSTormer_c23": "FakeSTormer",
    "ln":       "Ours (LN)",
    "lora":     "Ours (LoRA)",
    "gsd_lora": "Ours (GSD-LoRA)",
}
df = df.loc[ROW_ORDER]

# Dataset display sets
XD13 = ["UADFV","DFD-C23","DFDC","CDFv2","FFIW10K","KoDF","FAVC","DFDM-C23",
        "PGF","IDFv1","DSv1","DSv2","CDFv3"]
DISPLAY_XD13 = {"DFD-C23": "DFD", "DFDM-C23": "DFDM", "FFIW10K": "FFIW",
                "DSv1": "DSv1.1", "DSv2": "DSv2", "IDFv1": "IDFD"}
INDOMAIN = ["FF-DF-C23","FF-F2F-C23","FF-FS-C23","FF-NT-C23"]
DISPLAY_IND = {"FF-DF-C23": "FF-DF", "FF-F2F-C23": "FF-F2F",
               "FF-FS-C23": "FF-FS", "FF-NT-C23": "FF-NT"}

# --------------------------------------------------------------------------- #
# Figure 1: Method × dataset heatmap (cross-domain 13)
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10.5, 5.0))
M = df[XD13].values  # rows = methods, cols = datasets
labels_x = [DISPLAY_XD13.get(d, d) for d in XD13]
labels_y = [ROW_LABELS[m] for m in df.index]

cmap = LinearSegmentedColormap.from_list(
    "perf", [(0.0, "#7B003C"), (0.45, "#FFE082"), (1.0, "#2E7D32")])
im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=60, vmax=100)
ax.set_xticks(range(len(labels_x))); ax.set_xticklabels(labels_x, rotation=35, ha="right")
ax.set_yticks(range(len(labels_y))); ax.set_yticklabels(labels_y)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color="white" if v < 80 else "black", fontsize=7.5)
# Separator line between baselines and ours
ax.axhline(5.5, color="black", linewidth=1.2)
cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015)
cb.set_label("Video AUC (\\%)")
ax.set_title("Cross-dataset Video-AUC — 9 methods on the unified pipeline")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_PLOT / f"figS1_heatmap.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote figS1_heatmap.{pdf,png}")

# --------------------------------------------------------------------------- #
# Figure 2: Headline summary bar chart (MEAN-13 / MEAN-17 / MEAN-IND)
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8.5, 4.5))
x = np.arange(len(df))
width = 0.27
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ax.bar(x - width, df["MEAN_13"], width, label="MEAN-13 (cross-domain)", color=colors[0])
ax.bar(x,         df["MEAN_17"], width, label="MEAN-17 (broader Unseen)", color=colors[1])
ax.bar(x + width, df["MEAN_IND"], width, label="MEAN-IND (FF++ in-domain)", color=colors[2])

ax.set_xticks(x); ax.set_xticklabels([ROW_LABELS[m] for m in df.index],
                                    rotation=22, ha="right")
ax.set_ylabel("Video AUC (\\%)")
ax.set_ylim(75, 102)
ax.axhline(91.58, ls="--", lw=0.6, color="#888", label="GenD$^{\\mathrm{DINO}}$ MEAN-17 (91.58)")
ax.legend(loc="lower left", fontsize=8, ncol=2)
ax.grid(axis="y", linewidth=0.3, alpha=0.5)
ax.set_title("Headline performance summary")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_PLOT / f"figS3_summary.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote figS3_summary.{pdf,png}")

# --------------------------------------------------------------------------- #
# Figure 3: Per-dataset bar chart (cross-domain 13) for ALL methods
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(13.0, 4.5))
x = np.arange(len(XD13))
n = len(df)
w = 0.85 / n
# Strong distinguishing colour scheme: baselines blue family, ours red/green
COLOR_MAP = {
    "ForensicAdapter_v1":         "#6BAED6",
    "Effort-AIGI-Detection_FFpp": "#4292C6",
    "GenD_CLIP_L_14":             "#2171B5",
    "GenD_PE_L":                  "#08519C",
    "GenD_DINOv3_L":              "#08306B",
    "FakeSTormer_c23":            "#9ECAE1",
    "ln":       "#FFB44D",
    "lora":     "#FF6E33",
    "gsd_lora": "#C5202B",
}
for i, m in enumerate(df.index):
    vals = df.loc[m, XD13].values
    ax.bar(x + (i - n/2) * w + w/2, vals, w, label=ROW_LABELS[m],
           color=COLOR_MAP.get(m, "#888"))

ax.set_xticks(x); ax.set_xticklabels([DISPLAY_XD13.get(d, d) for d in XD13],
                                    rotation=30, ha="right")
ax.set_ylabel("Video AUC (\\%)")
ax.set_ylim(55, 102)
ax.legend(ncol=5, fontsize=7.2, loc="lower right")
ax.grid(axis="y", linewidth=0.3, alpha=0.4)
ax.set_title("Per-dataset Video AUC — cross-domain (13 datasets), all 9 methods")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_PLOT / f"figS2_per_dataset.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote figS2_per_dataset.{pdf,png}")

# --------------------------------------------------------------------------- #
# Figure 4: Δ(GSD-LoRA – GenD-DINOv3) per-dataset (17 cross-domain)
# --------------------------------------------------------------------------- #
XD17 = ["FF-FSH-C23","CDFv1","CDFv2","CDFv3","DeFo","DFD-C23","DSv1","DSv2",
        "DFDCP","DFDC","DFDM-C23","FAVC","FFIW10K","IDFv1","KoDF","PGF","UADFV"]
DISP17 = {**DISPLAY_XD13, "FF-FSH-C23": "FF-FSH"}
delta = (df.loc["gsd_lora", XD17] - df.loc["GenD_DINOv3_L", XD17]).sort_values()
fig, ax = plt.subplots(figsize=(9.0, 4.5))
colors_d = ["#C5202B" if v > 0 else "#3A6FB8" for v in delta]
ax.barh(range(len(delta)), delta.values, color=colors_d, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(delta)))
ax.set_yticklabels([DISP17.get(d, d) for d in delta.index])
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("$\\Delta$ Video AUC (Ours GSD-LoRA $-$ GenD$^{\\mathrm{DINO}}$, points)")
ax.set_title("Per-dataset advantage of GSD-LoRA over the strongest baseline")
for i, v in enumerate(delta.values):
    ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.grid(axis="x", linewidth=0.3, alpha=0.4)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_PLOT / f"figS4_delta_gsd_vs_gend.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Wrote figS4_delta_gsd_vs_gend.{pdf,png}")

print()
print(f"All figures written to {OUT_PLOT}")
