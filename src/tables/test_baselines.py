"""Tables 2, 3 and S2-S5 regenerate from comparison_master.csv, and that file's Ours rows match the run logs."""

import csv
import json
import re
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import pytest

from conftest import REPO, run_script
from paperfix import agrees, load
from tables.reproduce_tables import FF_TRAIN, MAIN_DIR, SEEDS, find_results

EMIT_PAPER = "src/tables/emit_paper_tables.py"
EMIT_SUPP = "src/tables/emit_supplementary_tables.py"
TEX_DIR = REPO / "results" / "baselines" / "tex"
MASTER = REPO / "results" / "baselines" / "comparison_master.csv"
PUBLISHED = REPO / "results" / "published" / "published_numbers.csv"

# the emitters' own dataset names and column orders
XD13 = ["UADFV", "DFD-C23", "DFDC", "CDFv2", "FFIW10K", "KoDF", "FAVC", "DFDM-C23",
        "PGF", "IDFv1", "DSv1", "DSv2", "CDFv3"]
XD17 = ["FF-FSH-C23", "CDFv1", "CDFv2", "CDFv3", "DeFo", "DFD-C23", "DSv1", "DSv2",
        "DFDCP", "DFDC", "DFDM-C23", "FAVC", "FFIW10K", "IDFv1", "KoDF", "PGF", "UADFV"]
IND = ["FF-DF-C23", "FF-F2F-C23", "FF-FS-C23", "FF-NT-C23"]
# the paper prints shorter names than the master file
PAPER_NAME = {"DFD-C23": "DFD", "FFIW10K": "FFIW", "DFDM-C23": "DFDM", "DSv1": "DSv1.1", "FF-FSH-C23": "FF-FSH",
              "FF-DF-C23": "FF-DF", "FF-F2F-C23": "FF-F2F", "FF-FS-C23": "FF-FS", "FF-NT-C23": "FF-NT"}
DETECTORS = ["ForensicAdapter_v1", "Effort-AIGI-Detection_FFpp", "GenD_CLIP_L_14", "GenD_PE_L", "GenD_DINOv3_L"]
SHORT = {"ForensicAdapter_v1": "FA", "Effort-AIGI-Detection_FFpp": "Effort", "GenD_CLIP_L_14": "GenD-CLIP",
         "GenD_PE_L": "GenD-PE", "GenD_DINOv3_L": "GenD-DINO", "FakeSTormer_c23": "FakeSTormer"}
METHODS = DETECTORS + ["FakeSTormer_c23", "ln", "lora", "gsd_lora"]

DELTA = re.compile(r"\$([+-]\d+\.\d+)\$")
VALUE = re.compile(r"\d+\.\d+")


def paper(name):
    return PAPER_NAME.get(name, name)


def latex_cells(line):
    """[(value, delta)] for the data cells of one LaTeX table row; value is None for a '---' cell."""
    body = line.replace("\\midrule", "").strip().rstrip("\\").strip()
    cells = [c.strip() for c in body.split(" & ")][1:]
    parsed = []
    for cell in cells:
        m = VALUE.search(cell)
        delta = DELTA.search(cell)
        parsed.append((m.group(0) if m else None, delta.group(1) if delta else None))
    return parsed


def data_rows(text):
    return [line for line in text.splitlines() if " & " in line]


def block(stdout, start):
    """The lines printed under the '=== <start>' banner, up to the next banner or blank line."""
    lines = stdout.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("=== " + start))
    out = []
    for line in lines[first + 1:]:
        if line.startswith("===") or not line.strip():
            break
        out.append(line)
    return out


def compare(table_id, fixture_rows, parsed, columns, with_delta=False, ties=None):
    """Compare parsed [(value, delta)] rows against the fixture; return (cells compared, mismatches)."""
    ties = ties or {}
    fixture = load(table_id)
    assert len(fixture_rows) == len(parsed), f"{table_id}: {len(parsed)} rows regenerated, {len(fixture_rows)} expected"
    compared, bad = 0, []
    for name, cells in zip(fixture_rows, parsed):
        assert len(cells) == len(columns), f"{table_id}/{name}: {len(cells)} cells, {len(columns)} columns"
        for col, (value, delta) in zip(columns, cells):
            printed = fixture[(name, col)]
            compared += 1
            if (table_id, name, col) in ties:
                value = repr(float(ties[(table_id, name, col)]))
            if not agrees(printed, float(value)):
                bad.append((table_id, name, col, printed, value))
            if with_delta and (name, col + " Δ") in fixture:
                compared += 1
                if (table_id, name, col + " Δ") in ties:
                    delta = repr(float(ties[(table_id, name, col + " Δ")]))
                if delta is None or not agrees(fixture[(name, col + " Δ")], float(delta)):
                    bad.append((table_id, name, col + " Δ", fixture[(name, col + " Δ")], delta))
    return compared, bad


def half_up(x):
    return float(Decimal(f"{float(x):.6f}").quantize(Decimal("0.01"), ROUND_HALF_UP))


def known_ties():
    """Cells recomputed from comparison_master.csv floats instead of the emitter's text.

    Exact x.xx5 values: the paper rounds half up; the emitter formats with Python rounding.
    """
    cm = pd.read_csv(MASTER, index_col=0)
    clip_mean = cm.loc["GenD_CLIP_L_14", IND].mean()
    fa_mean = cm.loc["ForensicAdapter_v1", IND].mean()
    with open(PUBLISHED, newline="") as handle:
        fa_pub_mean = next(float(r["value"]) for r in csv.DictReader(handle)
                           if (r["table"], r["row"], r["dataset"]) == ("S3", "ForensicAdapter_v1", "MEAN"))
    return {
        "table2": {("table2", "GenD-CLIP", "MEAN"): clip_mean},
        "tableS3": {("tableS3", "FA (ours)", "MEAN"): fa_mean,
                    ("tableS3", "FA (ours)", "MEAN \u0394"): half_up(fa_mean) - fa_pub_mean},
    }


KNOWN_TIES = known_ties


OURS = ("orthoflow_gtl_ln", "orthoflow_gtl_lora", "orthoflow_gtl_gsd_lora")


def log_rows(datasets, mean_key):
    """Ours rows (LN, LoRA, GSD-LoRA) from the run logs: unrounded 3-seed means, as the paper prints them."""
    rows = []
    for prefix in OURS:
        logs = seed_mean_per_dataset(prefix)
        values = [logs[d] for d in datasets]
        mean = logs[mean_key] if mean_key in logs else sum(values) / len(values)
        rows.append([(repr(v), None) for v in values + [mean]])
    return rows


def test_table2_all_rows():
    out = run_script(EMIT_PAPER)
    rows = [latex_cells(line) for line in block(out, "Table 2")][:6] + log_rows(IND, "MEAN_IND")
    names = ["Forensics Adapter", "Effort", "GenD-CLIP", "GenD-PE", "GenD-DINO", "FakeSTormer",
             "Ours (LN)", "Ours (LoRA)", "Ours (GSD-LoRA)"]
    compared, bad = compare("table2", names, rows, [paper(d) for d in IND] + ["MEAN"], ties=KNOWN_TIES()["table2"])
    assert compared == 45
    assert not bad, bad


def test_table3_all_rows():
    run_script(EMIT_SUPP)
    rows = [latex_cells(line) for line in data_rows((TEX_DIR / "supp_table_full17_all_methods.tex").read_text())][:6]
    rows += log_rows(XD17, "MEAN_17")
    names = ["Forensics Adapter", "Effort", "GenD-CLIP", "GenD-PE", "GenD-DINO", "FakeSTormer", "LN", "LoRA", "GSD-LoRA"]
    compared, bad = compare("table3", names, rows, [paper(d) for d in XD17] + ["MEAN"])
    assert compared == 162
    assert not bad, bad


def _s4_rows():
    """Table S4 from the published numbers and the master file: the paper has no emitter for it."""
    master = pd.read_csv(MASTER, index_col=0)
    with open(PUBLISHED, newline="") as handle:
        pub = [r for r in csv.DictReader(handle) if r["table"] == "S4"]
    spec = {
        "ForensicAdapter_v1": ("FA", "FA (orig.)"),
        "Effort-AIGI-Detection_FFpp": ("Effort", "Effort (orig.)"),
        "FakeSTormer_c23": ("FakeSTormer", "FakeSTormer (orig., T=4)"),
        "GenD_CLIP_L_14": ("GenD-CLIP", "GenD-CLIP (orig.)"),
        "GenD_PE_L": ("GenD-PE", "GenD-PE (orig.)"),
        "GenD_DINOv3_L": ("GenD-DINO", "GenD-DINO (orig.)"),
    }
    expected = {}
    for det, (short, orig_name) in spec.items():
        orig = {r["dataset"]: float(r["value"]) for r in pub if r["row"] == det and r["dataset"] != "MEAN"}
        typed_mean = next(r["value"] for r in pub if r["row"] == det and r["dataset"] == "MEAN")
        orig_mean = sum(orig.values()) / len(orig)
        assert agrees(typed_mean, orig_mean), (det, typed_mean, orig_mean)
        expected[orig_name] = {**{paper(d): v for d, v in orig.items()}, "MEAN": orig_mean}
        ours = {d: master.loc[det, d] for d in XD17}
        row = {paper(d): v for d, v in ours.items()}
        for d, v in orig.items():
            row[paper(d) + " Δ"] = ours[d] - v
        ours_mean = sum(ours[d] for d in orig) / len(orig)
        row["MEAN"] = ours_mean
        # the paper prints the MEAN delta as the difference of the two printed means
        row["MEAN Δ"] = float(Decimal(repr(float(ours_mean))).quantize(Decimal("0.01"), ROUND_HALF_UP)) - orig_mean
        expected[short + " (ours)"] = row
    return expected


def test_tables_S2_to_S5():
    fixture_s4 = load("tableS4")
    # S2 and S3: the emitter's stacked published / re-evaluated rows
    run_script(EMIT_SUPP)
    out = run_script(EMIT_PAPER)
    cross = data_rows((TEX_DIR / "supp_table_baseline_published_vs_reeval_cross.tex").read_text())
    names2 = [n for d in DETECTORS for n in (SHORT[d] + " (reprod.)", SHORT[d] + " (ours)")] + ["FST (ours)"]
    cols2 = [paper(d) for d in XD13] + ["MEAN"]
    c2, bad2 = compare("tableS2", names2, [latex_cells(r) for r in cross], cols2, with_delta=True)
    indomain = data_rows((TEX_DIR / "supp_table_baseline_published_vs_reeval_indomain.tex").read_text())
    names3 = [n for d in DETECTORS for n in (SHORT[d] + " (reprod.)", SHORT[d] + " (ours)")] + ["FakeSTormer (ours)"]
    cols3 = [paper(d) for d in IND] + ["MEAN"]
    c3, bad3 = compare("tableS3", names3, [latex_cells(r) for r in indomain], cols3, with_delta=True, ties=KNOWN_TIES()["tableS3"])
    # S4: every typed cell and every delta
    expected = _s4_rows()
    assert set(expected) <= {r for r, _ in fixture_s4}
    c4, bad4 = 0, []
    for (row, col), printed in fixture_s4.items():
        c4 += 1
        want = expected[row].get(col)
        if want is None or not agrees(printed, want):
            bad4.append(("tableS4", row, col, printed, want))
    # S5: the 13-dataset block of the headline emitter
    s5_names = ["FA", "Effort", "GenD-CLIP", "GenD-PE", "GenD-DINO", "FST", "LN", "LoRA", "GSD-LoRA"]
    cols5 = [paper(d) for d in XD13] + ["MEAN-13"]
    c5, bad5 = compare("tableS5", s5_names, [latex_cells(r) for r in block(out, "Table 1")][:6] + log_rows(XD13, "MEAN_13"), cols5)
    print(f"cells compared: S2 {c2}, S3 {c3}, S4 {c4}, S5 {c5}")
    assert (c2, c3, c4, c5) == (len(load("tableS2")), len(load("tableS3")), len(fixture_s4), 126)
    assert not (bad2 + bad3 + bad4 + bad5), bad2 + bad3 + bad4 + bad5


# ---- the master file's Ours rows against the run logs ----

def per_dataset(cm, variant):
    """The master file's per-dataset Video-AUC for one Ours variant, plus its MEAN_17 and MEAN_IND."""
    row = cm.loc[{"LN": "ln", "LoRA": "lora", "GSD-LoRA": "gsd_lora"}[variant]]
    cols = [c for c in cm.columns[2:] if not c.endswith("_std") and not c.startswith("MEAN")]
    out = {c: float(row[c]) for c in cols}
    out["MEAN_17"] = float(row["MEAN_17"])
    out["MEAN_IND"] = float(row["MEAN_IND"])
    return out


def per_seed_values(prefix):
    """Per-seed best_macro Video-AUC (x 100) for every dataset except FF-ORIGINAL-C23."""
    per_seed = []
    for seed in SEEDS:
        path = find_results(MAIN_DIR / f"{prefix}_{seed}", "best_macro")
        assert path is not None, f"no best_macro test results for {prefix}_{seed}"
        datasets = json.loads(path.read_text())["datasets"]
        per_seed.append({d: v["video_metrics"]["auc"] * 100 for d, v in datasets.items() if d != "FF-ORIGINAL-C23"})
    return per_seed


def seed_mean_per_dataset(prefix, rounded=False):
    """Mean over seeds of the best_macro test_results.json, per dataset (AUC x 100), optionally of 2 dp per-seed values."""
    per_seed = per_seed_values(prefix)
    if rounded:
        per_seed = [{d: round(v, 2) for d, v in s.items()} for s in per_seed]
    out = {d: sum(s[d] for s in per_seed) / len(per_seed) for d in per_seed[0]}
    cross = [d for d in out if d not in FF_TRAIN]
    out["MEAN_17"] = sum(out[d] for d in cross) / len(cross)
    out["MEAN_13"] = sum(out[d] for d in XD13) / len(XD13)
    out["MEAN_IND"] = sum(out[d] for d in IND) / len(IND)
    return out


def test_comparison_master_ours_rows_match_logs():
    """Provenance of the master file's Ours rows, column by column.

    LN and LoRA: the 17 cross-dataset columns and MEAN_17 equal the mean over seeds of per-seed values rounded to 2 dp.
    LN and LoRA in-domain FF columns, and every GSD-LoRA column, equal the unrounded log mean.
    The paper prints the unrounded log means, so no master value is more than 0.005 from them.
    """
    cm = pd.read_csv(MASTER, index_col=0)
    for variant, prefix in (("LN", "orthoflow_gtl_ln"), ("LoRA", "orthoflow_gtl_lora"),
                            ("GSD-LoRA", "orthoflow_gtl_gsd_lora")):
        master = per_dataset(cm, variant)
        rounded = seed_mean_per_dataset(prefix, rounded=True)
        exact = seed_mean_per_dataset(prefix)
        for key, value in master.items():
            seed_rounded = variant in ("LN", "LoRA") and key not in IND and key != "MEAN_IND"
            reference = rounded if seed_rounded else exact
            assert value == pytest.approx(reference[key], abs=1e-9), (variant, key, value - reference[key])
            assert abs(value - exact[key]) < 0.005, (variant, key, value - exact[key])
