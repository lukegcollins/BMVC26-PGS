"""Build training-dynamics figures for the experiments section + supplementary.

Replaces the two-subfigure `figure*` (lora_dynamics_aggregated.pdf + val_curves.pdf).
Reduces vertical footprint and avoids the floats-after-references problem.

Outputs (all to figures/):
  training_dynamics_combined.{pdf,png}     — 3-panel combined (a)+(b)+(c)
  training_dynamics_val_curve.{pdf,png}    — panel (c) only (for main paper)
  figS6_training_dynamics.{pdf,png}        — panels (a)+(b) side-by-side (Fig. S6)

Panels:
  (a) Peak validation epoch per variant (boxplot + per-seed scatter + μ labels)
  (b) Top-3 checkpoint span (plateau-width proxy) per variant
  (c) Macro validation Video-AUC trajectory across epochs (mean ± SEM, full range)

Inputs:
  results/training_dynamics/val_curves.csv
  results/training_dynamics/overfitting_summary.csv

Run:
  python build_training_dynamics_figure.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = REPO_ROOT / "results" / "training_dynamics"
OUT_DIR = REPO_ROOT / "figures"

VARIANTS = ["ln", "lora", "gsd_lora"]
LABELS = {"ln": "LN", "lora": "LoRA", "gsd_lora": "GSD-LoRA"}
# Muted fills for box faces, saturated edge/line colours for the val-curve traces.
# Matches the muted blue / orange / green of the original separate figures.
BOX_FILL = {"ln": "#9ec0db", "lora": "#f4c08a", "gsd_lora": "#a8d5a8"}
LINE_COLOR = {"ln": "#1f77b4", "lora": "#d97a2c", "gsd_lora": "#2ca02c"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def top3_span(grp: pd.DataFrame) -> int:
    """Span (max - min epoch) of the top-3 epochs by macro val Video-AUC."""
    top3 = grp.nlargest(3, "val_video_auc")
    return int(top3.epoch.max() - top3.epoch.min())


def plot_box_with_scatter(ax, df, value_col, ylabel, title):
    """Boxplot + per-seed scatter + μ annotation, in the style of the original figures."""
    positions = np.arange(len(VARIANTS), dtype=float)
    data = [df[df.variant == v][value_col].values for v in VARIANTS]
    means = [float(np.mean(vals)) for vals in data]

    bp = ax.boxplot(
        data, positions=positions, widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )
    for patch, v in zip(bp["boxes"], VARIANTS):
        patch.set_facecolor(BOX_FILL[v])
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)

    rng = np.random.RandomState(0)
    for i, (v, vals) in enumerate(zip(VARIANTS, data)):
        jitter = rng.normal(0, 0.05, len(vals))
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter, vals,
            color=BOX_FILL[v], edgecolors="black", linewidths=0.5, zorder=3, s=24,
        )

    # μ labels just to the right of each box, at the mean level.
    for i, m in enumerate(means):
        ax.annotate(
            rf"$\mu={m:.1f}$",
            xy=(i + 0.30, m),
            xytext=(3, 0), textcoords="offset points",
            va="center", ha="left", fontsize=8,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[v] for v in VARIANTS])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    # Right-side padding so the μ labels don't get clipped.
    ax.set_xlim(-0.55, len(VARIANTS) - 1 + 0.95)


def plot_val_curve(ax, val_macro, *, show_title=True):
    """Macro val Video-AUC vs epoch; mean ± SEM across seeds, line per variant."""
    for v in VARIANTS:
        sub = val_macro[val_macro.variant == v]
        agg = sub.groupby("epoch")["val_video_auc_pct"].agg(["mean", "sem"]).reset_index()
        ax.plot(
            agg.epoch, agg["mean"], label=LABELS[v],
            color=LINE_COLOR[v], linewidth=1.4, marker="o", markersize=2.6,
        )
        ax.fill_between(
            agg.epoch, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"],
            color=LINE_COLOR[v], alpha=0.18, linewidth=0,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro val Video-AUC (\\%)" if plt.rcParams["text.usetex"] else "Macro val Video-AUC (%)")
    if show_title:
        ax.set_title("(c) Validation trajectory", loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {pdf_path}")


def build_combined(summary, plateau, val_macro):
    """3-panel combined figure: (a) peak epoch, (b) plateau width, (c) val curve."""
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.6), constrained_layout=True)
    plot_box_with_scatter(
        axes[0], summary, value_col="peak_epoch",
        ylabel="Peak validation epoch", title="(a) Peak epoch",
    )
    plot_box_with_scatter(
        axes[1], plateau, value_col="top3_span",
        ylabel="Top-3 checkpoint span (epochs)", title="(b) Plateau width",
    )
    plot_val_curve(axes[2], val_macro)
    save(fig, "training_dynamics_combined")


def build_val_curve_only(val_macro):
    """Single-panel val trajectory for the main paper's side-by-side figure*."""
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    plot_val_curve(ax, val_macro, show_title=False)
    save(fig, "training_dynamics_val_curve")


def build_box_summary(summary, plateau):
    """Panels (a)+(b) side-by-side for the supplementary."""
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), constrained_layout=True)
    plot_box_with_scatter(
        axes[0], summary, value_col="peak_epoch",
        ylabel="Peak validation epoch", title="(a) Peak epoch",
    )
    plot_box_with_scatter(
        axes[1], plateau, value_col="top3_span",
        ylabel="Top-3 checkpoint span (epochs)", title="(b) Plateau width",
    )
    save(fig, "figS6_training_dynamics")


def main():
    summary = pd.read_csv(ANALYSIS / "overfitting_summary.csv")
    val = pd.read_csv(ANALYSIS / "val_curves.csv")

    val_macro = val[val.criterion == "macro"].copy()
    val_macro["val_video_auc_pct"] = val_macro["val_video_auc"] * 100.0

    plateau = (
        val_macro
        .groupby(["run", "variant", "seed"])
        .apply(top3_span, include_groups=False)
        .reset_index(name="top3_span")
    )

    print("\n=== Peak epoch by variant (mean ± std) ===")
    for v in VARIANTS:
        vals = summary[summary.variant == v].peak_epoch.values
        print(f"  {LABELS[v]:<10} {vals.tolist()}  mean={vals.mean():.2f}  std={vals.std(ddof=1):.2f}")

    print("\n=== Top-3 checkpoint span by variant (mean ± std) ===")
    for v in VARIANTS:
        vals = plateau[plateau.variant == v].top3_span.values
        print(f"  {LABELS[v]:<10} {vals.tolist()}  mean={vals.mean():.2f}  std={vals.std(ddof=1):.2f}")

    print()
    build_combined(summary, plateau, val_macro)
    build_val_curve_only(val_macro)
    build_box_summary(summary, plateau)


if __name__ == "__main__":
    main()