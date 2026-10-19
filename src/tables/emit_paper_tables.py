#!/usr/bin/env python3
r"""Generate the LaTeX rows for the paper's Table 1 (cross-dataset, 13 datasets
+ MEAN) and Table 2 (in-domain, 4 datasets + MEAN), with best/2nd/3rd ranking
markers automatically computed across the 9 methods.

Reads `results/baselines/comparison_master.csv`; writes under `results/baselines/tex/`:
  - tables_table1_rows.tex  (paste under Table 1 \midrule baseline block)
  - tables_table2_rows.tex
  - tables_supplementary_full17.tex  (full 17-dataset version for supplementary)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "baselines" / "tex"
df = pd.read_csv(ROOT / "results" / "baselines" / "comparison_master.csv", index_col=0)

XD13 = ["UADFV","DFD-C23","DFDC","CDFv2","FFIW10K","KoDF","FAVC","DFDM-C23",
        "PGF","IDFv1","DSv1","DSv2","CDFv3"]
XD17 = ["FF-FSH-C23","CDFv1","CDFv2","CDFv3","DeFo","DFD-C23","DSv1","DSv2",
        "DFDCP","DFDC","DFDM-C23","FAVC","FFIW10K","IDFv1","KoDF","PGF","UADFV"]
INDOMAIN = ["FF-DF-C23","FF-F2F-C23","FF-FS-C23","FF-NT-C23"]
LAST_CITE = {
    "ForensicAdapter_v1":         r"[\citeyear{cui2025forensics}]",
    "Effort-AIGI-Detection_FFpp": r"[\citeyear{yan2025orthosubspace}]",
    "GenD_CLIP_L_14":             r"[\citeyear{yermakov2025gend}]",
    "GenD_PE_L":                  r"[\citeyear{yermakov2025gend}]",
    "GenD_DINOv3_L":              r"[\citeyear{yermakov2025gend}]",
    "FakeSTormer_c23":            r"[\citeyear{fakestormer}]",
}


def fmt_cell(val: float, rank: int) -> str:
    """Wrap value with bold/underline/italic per rank (1/2/3)."""
    s = f"{val:.2f}"
    if rank == 1:
        return r"\textbf{" + s + "}"
    if rank == 2:
        return r"{\ul " + s + "}"
    if rank == 3:
        return r"\textit{" + s + "}"
    return s


def rank_column(values: pd.Series) -> dict[str, int]:
    """Return method_id -> rank (1 = best, etc.). Ties get same rank."""
    sorted_vals = values.sort_values(ascending=False)
    ranks: dict[str, int] = {}
    last_val = None
    rank = 0
    same_count = 0
    for mid, v in sorted_vals.items():
        if last_val is None or abs(v - last_val) > 1e-9:
            rank += 1 + same_count
            same_count = 0
        else:
            same_count += 1
        ranks[mid] = rank
        last_val = v
    return ranks


def emit_table_rows(datasets: list[str], mean_col: str, label_map: dict) -> str:
    """Return a multi-line LaTeX string with one row per method, with markers."""
    # First, compute ranks per column
    col_ranks = {ds: rank_column(df[ds]) for ds in datasets + [mean_col]}
    out_lines: list[str] = []
    for mid, row in df.iterrows():
        cells = []
        for ds in datasets:
            r = col_ranks[ds][mid]
            cells.append(fmt_cell(row[ds], r))
        # MEAN column with the same formatting
        r_mean = col_ranks[mean_col][mid]
        mean_cell = fmt_cell(row[mean_col], r_mean)
        # Method label
        method_label = label_map.get(mid, mid)
        if mid in LAST_CITE:
            method_label_with_cite = (
                f"\\textbf{{{method_label}}} {LAST_CITE[mid]}"
            )
        else:
            method_label_with_cite = f"\\textbf{{{method_label}}}"

        # Last data column wraps in \multicolumn{1}{l|}{...} to match paper style
        # (in Table 1, the column before MEAN gets the bar). We replicate that.
        if len(datasets) > 1:
            cells[-1] = r"\multicolumn{1}{l|}{" + cells[-1] + "}"

        line = method_label_with_cite + " & " + " & ".join(cells) + " & " + mean_cell + r" \\"
        out_lines.append(line)
    return "\n".join(out_lines)


label_map = {
    "ForensicAdapter_v1":         "ForAda",
    "Effort-AIGI-Detection_FFpp": "Effort",
    "GenD_CLIP_L_14":             r"GenD\textsuperscript{CLIP}",
    "GenD_PE_L":                  r"GenD\textsuperscript{PE}",
    "GenD_DINOv3_L":              r"GenD\textsuperscript{DINO}",
    "FakeSTormer_c23":            "FakeSTormer",
    "ln":       "Ours (LN)",
    "lora":     "Ours (LoRA)",
    "gsd_lora": "Ours (GSD-LoRA)",
}

# Table 1: 13 cross-dataset + MEAN_13
t1 = emit_table_rows(XD13, "MEAN_13", label_map)
# Split baselines / ours with \midrule
baseline_ids = ["ForensicAdapter_v1","Effort-AIGI-Detection_FFpp",
                "GenD_CLIP_L_14","GenD_PE_L","GenD_DINOv3_L","FakeSTormer_c23"]
ours_ids = ["ln","lora","gsd_lora"]
t1_lines = t1.split("\n")
t1_baseline_block = "\n".join(t1_lines[:len(baseline_ids)])
t1_ours_block = "\n".join(t1_lines[len(baseline_ids):])
t1_combined = t1_baseline_block + " \\midrule\n" + t1_ours_block
(OUT / "tables_table1_rows.tex").write_text(t1_combined + "\n")
print("=== Table 1 (13 cross-dataset) ===")
print(t1_combined)

# Table 2: 4 in-domain + MEAN_IND
t2 = emit_table_rows(INDOMAIN, "MEAN_IND", label_map)
t2_lines = t2.split("\n")
t2_baseline_block = "\n".join(t2_lines[:len(baseline_ids)])
t2_ours_block = "\n".join(t2_lines[len(baseline_ids):])
t2_combined = t2_baseline_block + " \\midrule\n" + t2_ours_block
(OUT / "tables_table2_rows.tex").write_text(t2_combined + "\n")
print()
print("=== Table 2 (4 in-domain) ===")
print(t2_combined)

# Supplementary: full 17-dataset cross-domain table (compact summary only)
print()
print("=== Headline summary (rounded) ===")
print(f"{'Method':<26} | {'MEAN-13':>8} | {'MEAN-17':>8} | {'MEAN-IND':>9}")
print("-" * 60)
for mid, row in df.iterrows():
    print(f"{label_map[mid]:<26} | {row['MEAN_13']:>8.2f} | {row['MEAN_17']:>8.2f} | {row['MEAN_IND']:>9.2f}")
