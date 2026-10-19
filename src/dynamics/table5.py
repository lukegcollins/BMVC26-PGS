"""Table 5: training stability across three seeds (mean over seeds), from the validation curves.

Per run, on the macro validation Video-AUC (25 logged epochs, 0-24):
  peak epoch and peak AUC; the least-squares slope from the peak through the final epoch (pp/epoch, any run with
  at least two points from the peak, otherwise NaN); and the trapezoidal time-average of the curve (endpoints
  half-weighted).
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CURVES = REPO / "results" / "training_dynamics" / "val_curves.csv"
VARIANTS = [("ln", "LN"), ("lora", "LoRA"), ("gsd_lora", "GSD-LoRA")]


def per_run(curves: pd.DataFrame) -> pd.DataFrame:
    macro = curves[curves["criterion"] == "macro"]
    out = []
    for run, sub in macro.groupby("run"):
        sub = sub.sort_values("epoch")
        epochs = sub["epoch"].to_numpy(float)
        auc = sub["val_video_auc"].to_numpy(float) * 100.0
        peak = int(np.argmax(auc))
        # Fitted from the peak through the final epoch with at least two points, as the paper describes.
        # LN seed 137 peaks at epoch 23, so its fit uses two points (-0.0100). A run that peaks at its final
        # epoch has no slope (NaN).
        slope = float(np.polyfit(epochs[peak:], auc[peak:], 1)[0]) if len(epochs) - peak >= 2 else float("nan")
        area = float(np.sum((auc[1:] + auc[:-1]) / 2 * np.diff(epochs)) / (epochs[-1] - epochs[0]))
        out.append(
            {
                "variant": sub["variant"].iloc[0],
                "peak_epoch": epochs[peak],
                "peak_auc": auc[peak],
                "slope": slope,
                "auc_under_curve": area,
            }
        )
    return pd.DataFrame(out)


def table5(curves: pd.DataFrame) -> dict[str, dict[str, float]]:
    runs = per_run(curves)
    means = runs.groupby("variant").mean(numeric_only=True)
    return {label: means.loc[key].to_dict() for key, label in VARIANTS}


def main() -> None:
    table = table5(pd.read_csv(CURVES))
    print("Table 5: training stability, mean over three seeds")
    print("Variant  Peak epoch  Peak val-AUC (%)  Post-peak slope (pp/epoch)  AUC-under-curve (%)")
    for label, v in table.items():
        print(f"{label}  {v['peak_epoch']:.1f}  {v['peak_auc']:.2f}  {v['slope']:.3f}  {v['auc_under_curve']:.2f}")


if __name__ == "__main__":
    main()
