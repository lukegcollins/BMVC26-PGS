#!/usr/bin/env python3
"""Parse every train.log and emit:
  - val_curves.csv  (long-form: run, epoch, criterion, video_auc, frame_auc, loss)
  - overfitting_summary.csv  (per-run peak/final val auc, gap, post-peak slope)
  - val_curves.{pdf,png}  (per-criterion mean over seeds per variant)

Used to validate the "GSD lets LoRA train longer without overfitting" claim.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results"
RUNS = ROOT / "logs/main"
OUT = ROOT / "training_dynamics"

# Two regex passes per epoch block:
VAL_LINE = re.compile(
    r"(DSv2-VAL|CDFv3-VAL|FF-FSH-C23-VAL|MACRO \(3 datasets\)):\s+"
    r"loss=([\d.]+)\s+.*?video_auc=([\d.]+)"
)
EPOCH_MARK = re.compile(r"Epoch (\d+) validation: video_auc_macro=[\d.]+")
TRAIN_LINE = re.compile(r"\[Train E(\d+)\] loss=([-\d.]+).*?video_auc=([\d.]+)")

CRIT_MAP = {
    "DSv2-VAL": "dsv2",
    "CDFv3-VAL": "cdfv3",
    "FF-FSH-C23-VAL": "ffsh",
    "MACRO (3 datasets)": "macro",
}
RUN_PAT = re.compile(r"^orthoflow_gtl_(?P<v>ln|lora|gsd_lora)_s(?P<s>\d+)$")

rows = []
train_rows = []
for run_dir in sorted(RUNS.iterdir()):
    m = RUN_PAT.match(run_dir.name)
    if not m: continue
    variant, seed = m["v"], int(m["s"])
    log = run_dir / "train.log"
    if not log.exists(): continue
    pending: dict[str, tuple[float, float]] = {}
    pending_train: tuple[int, float, float] | None = None
    with log.open("r", errors="ignore") as fh:
        for line in fh:
            vm = VAL_LINE.search(line)
            if vm:
                pending[CRIT_MAP[vm.group(1)]] = (float(vm.group(2)),
                                                   float(vm.group(3)))
                continue
            em = EPOCH_MARK.search(line)
            if em and pending:
                ep = int(em.group(1))
                for crit, (loss, vauc) in pending.items():
                    rows.append({
                        "run": run_dir.name, "variant": variant, "seed": seed,
                        "epoch": ep, "criterion": crit,
                        "val_loss": loss, "val_video_auc": vauc,
                    })
                pending = {}
                continue
            tm = TRAIN_LINE.search(line)
            if tm:
                train_rows.append({
                    "run": run_dir.name, "variant": variant, "seed": seed,
                    "epoch": int(tm.group(1)),
                    "train_loss": float(tm.group(2)),
                    "train_video_auc": float(tm.group(3)),
                })

df = pd.DataFrame(rows)
df_train = pd.DataFrame(train_rows)
df.to_csv(OUT / "val_curves.csv", index=False)
df_train.to_csv(OUT / "train_curves.csv", index=False)
print(f"Val rows: {len(df)}, train rows: {len(df_train)}")
print(f"Runs covered: {df['run'].nunique()}; variants: {df['variant'].unique()}")

# --------------------------------------------------------------------------- #
# Overfitting summary per run (macro criterion)
# --------------------------------------------------------------------------- #
summary = []
for run, sub in df[df["criterion"] == "macro"].groupby("run"):
    sub = sub.sort_values("epoch")
    peak_idx = sub["val_video_auc"].idxmax()
    peak_ep = int(sub.loc[peak_idx, "epoch"])
    peak_auc = float(sub.loc[peak_idx, "val_video_auc"])
    final_ep = int(sub["epoch"].max())
    final_auc = float(sub.loc[sub["epoch"] == final_ep, "val_video_auc"].iloc[0])
    overfit_gap = peak_auc - final_auc
    # Plateau: epochs where val AUC stays within 0.5 pt of peak
    near_peak = sub[sub["val_video_auc"] >= peak_auc - 0.005]
    plateau_width = int(near_peak["epoch"].max() - near_peak["epoch"].min()) + 1
    # Post-peak slope: linear regression of val AUC vs epoch from peak to end
    post = sub[sub["epoch"] >= peak_ep]
    if len(post) >= 3:
        x = post["epoch"].values.astype(float)
        y = post["val_video_auc"].values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0]) * 100  # in percentage points / epoch
    else:
        slope = float("nan")
    summary.append({
        "run": run,
        "variant": sub["variant"].iloc[0],
        "seed": int(sub["seed"].iloc[0]),
        "peak_epoch": peak_ep,
        "peak_macro_auc": round(peak_auc * 100, 3),
        "final_epoch": final_ep,
        "final_macro_auc": round(final_auc * 100, 3),
        "overfit_gap_pts": round(overfit_gap * 100, 3),
        "plateau_within_0.5pt_epochs": plateau_width,
        "post_peak_slope_pts_per_epoch": round(slope, 4),
    })
summary_df = pd.DataFrame(summary).sort_values(["variant", "seed"])
summary_df.to_csv(OUT / "overfitting_summary.csv", index=False)
print()
print("=== Overfitting summary (macro criterion) ===")
print(summary_df.to_string(index=False))

print()
print("=== Variant aggregates (mean ± std across seeds) ===")
agg = summary_df.groupby("variant").agg(
    n=("run", "count"),
    peak_epoch_mean=("peak_epoch", "mean"),
    peak_epoch_std=("peak_epoch", "std"),
    peak_auc_mean=("peak_macro_auc", "mean"),
    final_auc_mean=("final_macro_auc", "mean"),
    overfit_gap_mean=("overfit_gap_pts", "mean"),
    overfit_gap_std=("overfit_gap_pts", "std"),
    plateau_mean=("plateau_within_0.5pt_epochs", "mean"),
    post_slope_mean=("post_peak_slope_pts_per_epoch", "mean"),
)
print(agg.round(3).to_string())

# --------------------------------------------------------------------------- #
# Plot: per-epoch macro val AUC, averaged over seeds per variant
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9.0, 4.5))
mac = df[df["criterion"] == "macro"].copy()
mac["val_video_auc_pct"] = mac["val_video_auc"] * 100
color_map = {"ln": "#377EB8", "lora": "#FF7F00", "gsd_lora": "#C5202B"}
label_map = {"ln": "LN", "lora": "LoRA", "gsd_lora": "GSD-LoRA"}
for v in ["ln", "lora", "gsd_lora"]:
    sub = mac[mac["variant"] == v]
    by_epoch = sub.groupby("epoch")["val_video_auc_pct"].agg(["mean", "std", "count"])
    by_epoch["sem"] = by_epoch["std"] / np.sqrt(by_epoch["count"])
    ax.plot(by_epoch.index, by_epoch["mean"], color=color_map[v],
            label=label_map[v], linewidth=1.8, marker="o", markersize=3.5)
    ax.fill_between(by_epoch.index,
                    by_epoch["mean"] - by_epoch["sem"],
                    by_epoch["mean"] + by_epoch["sem"],
                    color=color_map[v], alpha=0.15)
ax.set_xlabel("Epoch")
ax.set_ylabel("Macro validation Video-AUC (\\%)")
ax.set_title("Validation trajectory across training (mean ± SEM over 3 seeds)")
ax.set_xlim(-0.5, 25)
ax.set_ylim(40, 92)
ax.grid(linewidth=0.3, alpha=0.5)
ax.legend(loc="lower right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(REPO / "figures" / f"val_curves.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nWrote val_curves.{pdf,png}")
