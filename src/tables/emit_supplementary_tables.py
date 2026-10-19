#!/usr/bin/env python3
"""Emit LaTeX table rows for the supplementary document.

Produces three artefacts:
  * supp_table_baseline_published_vs_reeval_cross.tex  -- 13-dataset side-by-side
  * supp_table_baseline_published_vs_reeval_indomain.tex -- 4-dataset side-by-side
  * supp_table_full17_all_methods.tex                  -- full 17-dataset cross-domain
                                                          MEAN for all 9 methods

Format conventions match the main paper (small font, narrow column spacing).
"""

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "baselines" / "tex"
df = pd.read_csv(ROOT / "results" / "baselines" / "comparison_master.csv", index_col=0)

# Third-party reproduced values (GenD pipeline), read from results/published/published_numbers.csv.
# (FakeSTormer has no comparable published table; it is emitted as a single re-evaluated row.)
def _published(table: str) -> dict:
    out: dict = {}
    with open(ROOT / "results" / "published" / "published_numbers.csv", newline="") as handle:
        for rec in csv.DictReader(handle):
            if rec["table"] == table and rec["dataset"] != "MEAN":
                out.setdefault(rec["row"], {})[rec["dataset"]] = float(rec["value"])
    return out


PUBLISHED_T1 = _published("S2")
PUBLISHED_T2 = _published("S3")
DISP_LBL = {
    "ForensicAdapter_v1":         "ForAda",
    "Effort-AIGI-Detection_FFpp": "Effort",
    "GenD_CLIP_L_14":             r"GenD\textsuperscript{CLIP}",
    "GenD_PE_L":                  r"GenD\textsuperscript{PE}",
    "GenD_DINOv3_L":              r"GenD\textsuperscript{DINO}",
    "FakeSTormer_c23":            "FakeSTormer",
}

XD13 = ["UADFV","DFD-C23","DFDC","CDFv2","FFIW10K","KoDF","FAVC","DFDM-C23",
        "PGF","IDFv1","DSv1","DSv2","CDFv3"]
INDOMAIN = ["FF-DF-C23","FF-F2F-C23","FF-FS-C23","FF-NT-C23"]
DELTA_FMT = lambda x: r"\textcolor{red!70!black}{\textit{$" + f"{x:+.2f}" + r"$}}" if x < 0 \
                else r"\textcolor{darkgreen}{\textit{$" + f"{x:+.2f}" + r"$}}"


def emit_dual_row(det_id: str, datasets: list[str], pub_dict: dict, mean_col_pub: float) -> str:
    """Two stacked rows per detector: published / re-evaluated, with deltas inline."""
    out = []
    re_eval = df.loc[det_id]
    label = DISP_LBL.get(det_id, det_id)
    pub_vals = [pub_dict[d] for d in datasets]
    re_vals  = [re_eval[d]  for d in datasets]
    pub_mean = sum(pub_vals) / len(pub_vals)
    re_mean  = sum(re_vals)  / len(re_vals)
    # Row 1: published
    pub_cells = [f"{v:.2f}" for v in pub_vals] + [f"{pub_mean:.2f}"]
    out.append(r"\textbf{" + label + r"} (pub.) & " + " & ".join(pub_cells) + r" \\")
    # Row 2: re-evaluated + delta
    re_cells = []
    for pv, rv in zip(pub_vals, re_vals):
        delta = rv - pv
        re_cells.append(f"{rv:.2f} {DELTA_FMT(delta)}")
    delta_mean = re_mean - pub_mean
    re_cells.append(f"{re_mean:.2f} {DELTA_FMT(delta_mean)}")
    out.append(r"\textbf{" + label + r"} (ours) & " + " & ".join(re_cells) + r" \\")
    return "\n".join(out)


# --- Table A: 13-dataset side-by-side ---
order = ["ForensicAdapter_v1","Effort-AIGI-Detection_FFpp","GenD_CLIP_L_14",
         "GenD_PE_L","GenD_DINOv3_L"]
rows_t1 = []
for det_id in order:
    pub_vals = [PUBLISHED_T1[det_id][d] for d in XD13]
    pub_mean = sum(pub_vals) / len(pub_vals)
    rows_t1.append(emit_dual_row(det_id, XD13, PUBLISHED_T1[det_id], pub_mean))
# FakeSTormer has no published — emit single row
fs = df.loc["FakeSTormer_c23"]
fs_cells = [f"{fs[d]:.2f}" for d in XD13] + [f"{sum(fs[d] for d in XD13)/len(XD13):.2f}"]
rows_t1.append(r"\textbf{FakeSTormer} (ours) & " + " & ".join(fs_cells) + r" \\")
(OUT / "supp_table_baseline_published_vs_reeval_cross.tex").write_text(
    "\n\\midrule\n".join(rows_t1) + "\n"
)

# --- Table B: 4 in-domain side-by-side ---
rows_t2 = []
for det_id in order:
    pub_vals = [PUBLISHED_T2[det_id][d] for d in INDOMAIN]
    pub_mean = sum(pub_vals) / len(pub_vals)
    rows_t2.append(emit_dual_row(det_id, INDOMAIN, PUBLISHED_T2[det_id], pub_mean))
fs_cells = [f"{fs[d]:.2f}" for d in INDOMAIN] + [f"{sum(fs[d] for d in INDOMAIN)/len(INDOMAIN):.2f}"]
rows_t2.append(r"\textbf{FakeSTormer} (ours) & " + " & ".join(fs_cells) + r" \\")
(OUT / "supp_table_baseline_published_vs_reeval_indomain.tex").write_text(
    "\n\\midrule\n".join(rows_t2) + "\n"
)

# --- Table C: Full 17-dataset all 9 methods, single-row each ---
XD17 = ["FF-FSH-C23","CDFv1","CDFv2","CDFv3","DeFo","DFD-C23","DSv1","DSv2",
        "DFDCP","DFDC","DFDM-C23","FAVC","FFIW10K","IDFv1","KoDF","PGF","UADFV"]
LBL_OURS = {"ln":"Ours (LN)", "lora":"Ours (LoRA)", "gsd_lora":"Ours (GSD-LoRA)"}
all_methods = order + ["FakeSTormer_c23", "ln", "lora", "gsd_lora"]
rows_t3 = []
for mid in all_methods:
    row = df.loc[mid]
    lbl = DISP_LBL.get(mid) or LBL_OURS.get(mid, mid)
    cells = [f"{row[d]:.2f}" for d in XD17] + [f"{row['MEAN_17']:.2f}"]
    rows_t3.append(r"\textbf{" + lbl + r"} & " + " & ".join(cells) + r" \\")
(OUT / "supp_table_full17_all_methods.tex").write_text("\n".join(rows_t3) + "\n")

print("Wrote:")
print(" -", OUT / "supp_table_baseline_published_vs_reeval_cross.tex")
print(" -", OUT / "supp_table_baseline_published_vs_reeval_indomain.tex")
print(" -", OUT / "supp_table_full17_all_methods.tex")
