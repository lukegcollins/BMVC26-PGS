#!/usr/bin/env python3
"""Reproduce the camera-ready paper's results tables from the shipped
per-checkpoint test_results.json files.

Usage:
    python3 reproduce_tables.py            # every panel below
    python3 reproduce_tables.py indomain   # main Table 2 — in-domain FF++
    python3 reproduce_tables.py cross      # main Table 3 — cross-domain MEAN-17
    python3 reproduce_tables.py policy     # supp Table S6 — checkpoint policy
    python3 reproduce_tables.py ablations  # main Table 6 — single-framework ablation
    python3 reproduce_tables.py breakdown  # supp Tables S13-S16 (App. K) — full
                                           # per-run per-dataset breakdown
    python3 reproduce_tables.py sweeps     # supp Table S12 (App. I) —
                                           # hyperparameter sensitivity sweeps

Notes on checkpoint selection:
    Some runs have two `*_best_macro/` subdirectories (e.g. epoch03 and
    epoch09 for GSD-LoRA s137). This happens when the validation macro
    score has multiple local maxima during training. The canonical
    checkpoint is the one closest to that configuration's peak epoch
    (Table 5, main paper: LN ≈ 16.7, LoRA ≈ 3.3, GSD-LoRA ≈ 5.7). In
    practice this is always the earliest of the candidate dirs, which is
    what this script picks; the spurious later ones are retained for
    diagnostic transparency only.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parents[2] / "results"
MAIN_DIR = HERE / "logs" / "main"
ABLATION_DIR = HERE / "logs" / "ablations"
SWEEPS_DIR = HERE / "logs" / "ablations" / "sweeps"

FF_TRAIN = {
    "FF-ORIGINAL-C23", "FF-DF-C23", "FF-F2F-C23", "FF-FS-C23", "FF-NT-C23"
}
CONFIGS = {
    "LN":       "orthoflow_gtl_ln",
    "LoRA":     "orthoflow_gtl_lora",
    "GSD-LoRA": "orthoflow_gtl_gsd_lora",
}
SEEDS = ["s42", "s137", "s1729"]
POLICIES = ["best_macro", "best_dsv2", "best_cdfv3", "best_ffsh"]


def resolve(test_dir: Path, name: str) -> Path:
    """Follow ALIAS_OF if present."""
    d = test_dir / name
    alias = d / "ALIAS_OF"
    if alias.exists():
        target = alias.read_text().strip()
        return test_dir / target
    return d


def find_results(run_dir: Path, policy: str):
    test = run_dir / "test"
    if not test.is_dir():
        return None
    matches = sorted(d.name for d in test.iterdir() if d.name.endswith("_" + policy))
    if not matches:
        return None
    p = resolve(test, matches[0]) / "test_results.json"
    return p if p.exists() else None


def mean17(p: Path) -> float:
    ds = json.loads(p.read_text())["datasets"]
    aucs = [ds[d]["video_metrics"]["auc"] * 100 for d in ds if d not in FF_TRAIN]
    return sum(aucs) / len(aucs)


def in_domain(p: Path) -> dict:
    ds = json.loads(p.read_text())["datasets"]
    return {d: ds[d]["video_metrics"]["auc"] * 100 for d in ds if d in FF_TRAIN}


def policy_table():
    print("\n=== Supp. Table S6 (App. E): MEAN-17 under each checkpoint-selection policy ===\n")
    label = {"best_macro": "macro-best (default)",
             "best_dsv2": "DSv2-VAL only",
             "best_cdfv3": "CDFv3-VAL only",
             "best_ffsh": "FF-FSH-VAL only"}
    header = f"{'Policy':22} {'LN':>14} {'LoRA':>14} {'GSD-LoRA':>14}"
    print(header)
    print("-" * len(header))
    for policy in POLICIES:
        row = [label[policy]]
        for cfg_prefix in CONFIGS.values():
            vals = []
            for seed in SEEDS:
                p = find_results(MAIN_DIR / f"{cfg_prefix}_{seed}", policy)
                if p:
                    vals.append(mean17(p))
            row.append(f"{mean(vals):.2f} ± {stdev(vals):.2f}" if len(vals) == 3 else f"N={len(vals)}")
        print(f"{row[0]:22} {row[1]:>14} {row[2]:>14} {row[3]:>14}")


def cross_table():
    print("\n=== Main Table 3: cross-domain Video-AUC (macro-best, n=3 seeds) ===\n")
    print(f"{'Config':12} {'MEAN-17':>10}")
    print("-" * 24)
    for name, prefix in CONFIGS.items():
        vals = []
        for seed in SEEDS:
            p = find_results(MAIN_DIR / f"{prefix}_{seed}", "best_macro")
            if p:
                vals.append(mean17(p))
        print(f"{name:12} {mean(vals):>6.2f} ± {stdev(vals):.2f}")


def indomain_table():
    print("\n=== Main Table 2: in-domain FF++ Video-AUC (macro-best, n=3 seeds) ===\n")
    rows = {}
    for name, prefix in CONFIGS.items():
        per_set = {}
        for seed in SEEDS:
            p = find_results(MAIN_DIR / f"{prefix}_{seed}", "best_macro")
            if p:
                d = in_domain(p)
                for k, v in d.items():
                    per_set.setdefault(k, []).append(v)
        rows[name] = {k: (mean(vs), stdev(vs) if len(vs) > 1 else 0.0) for k, vs in per_set.items()}
    header = f"{'Config':12}" + "".join(f"{d:>16}" for d in sorted(FF_TRAIN))
    print(header)
    print("-" * len(header))
    for name, vals in rows.items():
        line = f"{name:12}"
        for d in sorted(FF_TRAIN):
            m, s = vals.get(d, (float('nan'), 0.0))
            line += f"{m:>10.2f}±{s:.2f}"
        print(line)


CROSS17_ORDER = [
    "FF-FSH-C23", "CDFv1", "CDFv2", "CDFv3", "DeFo", "DFD-C23",
    "DSv1", "DSv2", "DFDCP", "DFDC", "DFDM-C23", "FAVC",
    "FFIW10K", "IDFv1", "KoDF", "PGF", "UADFV",
]
CROSS17_SHORT = ["FFSH", "CDFv1", "CDFv2", "CDFv3", "DeFo", "DFD",
                 "DSv1", "DSv2", "DFDCP", "DFDC", "DFDM", "FAVC",
                 "FFIW", "IDFv1", "KoDF", "PGF", "UADFV"]


def breakdown_table():
    """Print supp App. K (Tables S13-S16) full per-run breakdown, all 4 criteria."""
    print("\n=== Supp. Tables S13-S16 (App. K): full per-run per-dataset breakdown ===\n")
    POLICY_LABELS = {"best_macro": "macro-best (default)",
                     "best_dsv2":  "DSv2-VAL only",
                     "best_cdfv3": "CDFv3-VAL only",
                     "best_ffsh":  "FF-FSH-VAL only"}
    for policy in POLICIES:
        print(f"--- Criterion: {POLICY_LABELS[policy]} ---")
        header = f"{'Run':28}" + "".join(f"{s:>7}" for s in CROSS17_SHORT) + f"{'Unseen':>9}"
        print(header)
        for cfg_name, cfg_prefix in CONFIGS.items():
            per_seed_aucs = []
            for seed in SEEDS:
                p = find_results(MAIN_DIR / f"{cfg_prefix}_{seed}", policy)
                if not p: continue
                ds = json.loads(p.read_text())["datasets"]
                aucs = [ds[k]["video_metrics"]["auc"] * 100 for k in CROSS17_ORDER]
                per_seed_aucs.append(aucs)
                m17 = sum(aucs) / len(aucs)
                label = f"{cfg_name} ({seed})"
                print(f"{label:28}" + "".join(f"{a:>7.2f}" for a in aucs) + f"{m17:>9.2f}")
            if len(per_seed_aucs) == 3:
                means = [mean(per_seed_aucs[s][i] for s in range(3)) for i in range(17)]
                sds   = [stdev(per_seed_aucs[s][i] for s in range(3)) for i in range(17)]
                u_per = [sum(per_seed_aucs[s][i] for i in range(17)) / 17 for s in range(3)]
                label = f"{cfg_name} mean (n=3)"
                print(f"{label:28}" + "".join(f"{means[i]:>4.2f}±{sds[i]:.2f}" for i in range(17)) + f"  {mean(u_per):.2f}±{stdev(u_per):.2f}")
            print()
        print()


SWEEP_PANELS = [
    ("Hinge margin m (LN tier; default 1.5)", [
        ("m=0.5", "sweep_m_0p5"),
        ("m=1.0", "sweep_m_1p0"),
        ("m=2.0", "sweep_m_2p0"),
        ("m=3.0", "sweep_m_3p0"),
    ]),
    ("Small-transform penalty lambda_small (LN tier; default 0.2)", [
        ("lambda=0.05", "sweep_lambda_small_0p05"),
        ("lambda=0.5",  "sweep_lambda_small_0p5"),
    ]),
    ("GTL mixing constant c (LN tier; default 0.01)", [
        ("c=0.001", "sweep_c_0p001"),
        ("c=0.1",   "sweep_c_0p1"),
    ]),
    ("GSD concentration / coherence weight eta (GSD-LoRA tier; default 5e-3)", [
        ("eta=1e-3", "sweep_eta_1e-3"),
        ("eta=2e-2", "sweep_eta_2e-2"),
    ]),
]


def sweeps_table():
    print("\n=== Supp. Table S12 (App. I): hyperparameter sensitivity sweeps ===\n")
    print("Single seed (42), 10-epoch budget (except sweep_eta_1e-3, see supp).")
    print(f"{'Value':16} {'Epoch':>6} {'FF-FSH':>9} {'MEAN-17':>9}")
    for panel_label, points in SWEEP_PANELS:
        print(f"\n-- {panel_label}")
        for label, suffix in points:
            run = SWEEPS_DIR / f"orthoflow_gtl_{suffix}_s42"
            test = run / "test"
            if not test.is_dir():
                print(f"{label:16}   missing: {test}")
                continue
            macro = next(d for d in sorted(test.iterdir()) if d.name.endswith("_best_macro"))
            ds = json.loads((macro / "test_results.json").read_text())["datasets"]
            ffsh = ds["FF-FSH-C23"]["video_metrics"]["auc"] * 100
            m17  = mean(ds[k]["video_metrics"]["auc"] * 100 for k in CROSS17_ORDER)
            print(f"{label:16} {macro.name.split('_')[0]:>6} {ffsh:>9.2f} {m17:>9.2f}")


# (display, run_prefix, root): root 'A' = logs/ablations, 'M' = logs/main.
# Mirrors main-paper Table 6 (single-framework ablation) + the supp
# component-sensitivity rows, all read from the shipped logs.
TABLE4_ROWS = [
    ("Frozen + linear probe",              "orthoflow_gtl_frozen_probe", "A"),
    ("LN tuning only (no geometric)",      "orthoflow_gtl_ln_no_geom",   "A"),
    ("  + OPA",                            "orthoflow_gtl_ln_opa_only",  "A"),
    ("  + OPA + uniformity",               "orthoflow_gtl_ln_opa_unif",  "A"),
    ("  + OPA + uniformity + GTL (Full method)", "orthoflow_gtl_ln",           "M"),
    ("  Full, no hard RR negatives",       "orthoflow_gtl_ln_no_hardrr", "A"),
    ("  Full, no small-transform penalty", "orthoflow_gtl_ln_no_small",  "A"),
    ("  GSD-LoRA, per-video vbar",         "orthoflow_gtl_gsd_lora_vbar","A"),
    ("LoRA + MLP (no geometric)",          "orthoflow_gtl_lora_no_geom", "A"),
    ("  + OPA + uniformity",               "orthoflow_gtl_lora_no_gtl",  "A"),
    ("  + OPA + uniformity + GTL",          "orthoflow_gtl_lora",         "M"),
    ("  + OPA + uniformity + GTL + GSD (GSD-LoRA)", "orthoflow_gtl_gsd_lora",     "M"),
]


def _macro_ckpt(run_dir: Path):
    """Pick the macro-best test dir, handling both naming conventions
    (epochNN_best_macro and the Lightning epoch=NN-video_auc=X form);
    earliest epoch wins (the genuine val peak)."""
    test = run_dir / "test"
    if not test.is_dir():
        return None
    cands = [d for d in test.iterdir()
             if d.name.endswith("_best_macro") or (d.name.startswith("epoch=") and "video_auc=" in d.name)]
    if not cands:
        return None
    def ep(d: Path) -> int:
        nm = d.name
        try:
            return int(nm.split("epoch=")[1].split("-")[0]) if nm.startswith("epoch=") \
                else int(nm.split("epoch")[1].split("_")[0])
        except Exception:
            return 9999
    return min(cands, key=ep)


def ablations_table():
    print("\n=== Main Table 6: single-framework ablation (Video-AUC %, n=3 except vbar) ===\n")
    print(f"{'Configuration':44} {'n':>2} {'DSv2':>6} {'CDFv3':>6} {'MEAN-17':>8}")
    print("-" * 70)
    for display, prefix, root in TABLE4_ROWS:
        base = ABLATION_DIR if root == "A" else MAIN_DIR
        per_ds, n = {}, 0
        for seed in SEEDS:
            ck = _macro_ckpt(base / f"{prefix}_{seed}")
            if ck is None:
                continue
            j = ck / "test_results.json"
            if not j.exists():
                continue
            ds = json.loads(j.read_text())["datasets"]
            n += 1
            for k, blk in ds.items():
                per_ds.setdefault(k, []).append(blk["video_metrics"]["auc"] * 100)
        if n == 0:
            print(f"{display:44} {'--':>2}")
            continue
        avg = {k: mean(v) for k, v in per_ds.items()}
        m17 = mean(avg[k] for k in CROSS17_ORDER if k in avg)
        dsv2 = avg.get("DSv2", float("nan"))
        cdfv3 = avg.get("CDFv3", float("nan"))
        print(f"{display:44} {n:>2} {dsv2:>6.2f} {cdfv3:>6.2f} {m17:>8.2f}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in ("all", "indomain"):
        indomain_table()
    if arg in ("all", "cross"):
        cross_table()
    if arg in ("all", "policy"):
        policy_table()
    if arg in ("all", "ablations"):
        ablations_table()
    if arg in ("all", "breakdown"):
        breakdown_table()
    if arg in ("all", "sweeps"):
        sweeps_table()


if __name__ == "__main__":
    main()
