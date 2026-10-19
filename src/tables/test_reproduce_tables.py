"""Each panel of reproduce_tables.py must regenerate the numbers printed in the paper."""

import importlib.util
import re
from pathlib import Path

import pytest

from conftest import REPO, run_script
from paperfix import agrees, load, rows

SCRIPT = "src/tables/reproduce_tables.py"
PM = re.compile(r"(-?\d+\.\d{2})\s*±\s*(\d+\.\d{2})")

COLS6 = {"DSv2": 1, "CDFv3": 2, "MEAN": 3}
COLS3 = {"MEAN": 0}
# the script prints DF, F2F, FS, NT, ORIGINAL; the paper's MEAN column is the ORIGINAL (last) column
COLS2 = {"FF-DF": 0, "FF-F2F": 1, "FF-FS": 2, "FF-NT": 3, "MEAN": 4}
CONFIGS = {"LN": 0, "LoRA": 1, "GSD-LoRA": 2}
CROSS17 = ["FFSH", "CDFv1", "CDFv2", "CDFv3", "DeFo", "DFD", "DSv1", "DSv2", "DFDCP", "DFDC",
           "DFDM", "FAVC", "FFIW", "IDFv1", "KoDF", "PGF", "UADFV", "Unseen"]
POLICY_ROWS = {"macro-best (default)", "DSv2-VAL only", "CDFv3-VAL only", "FF-FSH-VAL only"}


def _module():
    spec = importlib.util.spec_from_file_location("reproduce_tables", REPO / SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pm_lines(text):
    """{label: [means]} for lines whose cells print as mean±sd (glued, so rows() cannot read them)."""
    out = {}
    for line in text.splitlines():
        cells = PM.findall(line)
        if cells:
            label = line[: PM.search(line).start()].strip()
            out[label] = [float(m) for m, _ in cells]
    return out


def test_script_exposes_helpers():
    mod = _module()
    assert callable(mod.find_results) and callable(mod.mean17)
    p = mod.find_results(mod.MAIN_DIR / "orthoflow_gtl_gsd_lora_s42", "best_macro")
    assert p is not None and p.exists()
    assert agrees("92.48", mod.mean17(p))


def test_indomain_panel_matches_table2():
    got = _pm_lines(run_script(SCRIPT, "indomain"))
    fx = {k: v for k, v in load("table2").items() if k[0].startswith("Ours")}
    assert len(fx) == 15
    for (row, col), printed in fx.items():
        cfg = row[row.index("(") + 1 : -1]
        vals = got[cfg]
        assert agrees(printed, vals[COLS2[col]]), (row, col)


def test_cross_panel_matches_table3_ours_mean():
    got = _pm_lines(run_script(SCRIPT, "cross"))
    fx = {k: v for k, v in load("table3").items() if k[0] in CONFIGS and k[1] == "MEAN"}
    assert len(fx) == 3
    for (row, col), printed in fx.items():
        assert agrees(printed, got[row][COLS3[col]]), (row, col)


def test_policy_panel_matches_tableS6():
    got = _pm_lines(run_script(SCRIPT, "policy"))
    fx = load("tableS6")
    assert len(fx) == 12
    for (row, col), printed in fx.items():
        assert agrees(printed, got[row][CONFIGS[col]]), (row, col)


def _ordered_rows(text):
    """[(label, values)] in print order, keeping repeated labels."""
    out = []
    for line in text.splitlines():
        parsed = rows(line)
        out.extend(parsed.items())
    return out


def test_ablations_panel_matches_table6():
    ordered = _ordered_rows(run_script(SCRIPT, "ablations"))
    fx = load("table6")
    assert len(fx) == 27
    for (row, col), printed in fx.items():
        tier = re.search(r"\((LN|LoRA) tier\)$", row)
        label = row[: tier.start()].strip() if tier else row
        matches = [v for k, v in ordered if k == label]
        values = matches[{"LN": 0, "LoRA": 1}[tier.group(1)]] if tier else matches[0]
        assert agrees(printed, values[COLS6[col]]), (row, col)


S11_MAP = {
    "+ OPA + uniformity + GTL (Full method, anchor)": "+ OPA + uniformity + GTL (Full method)",
    "Full method, no hard RR negatives": "Full, no hard RR negatives",
    "Full method, no small-transform penalty": "Full, no small-transform penalty",
    "GSD-LoRA, per-video vbar instead of per-pair v": "GSD-LoRA, per-video vbar",
}


def test_ablations_panel_matches_tableS11():
    text = run_script(SCRIPT, "ablations")
    got = rows(text)
    brk = _breakdown_sections(run_script(SCRIPT, "breakdown"))["macro-best (default)"]
    fx = load("tableS11")
    assert len(fx) == 15
    for (row, col), printed in fx.items():
        if row == "GSD-LoRA, per-pair v (anchor)":
            run = brk["GSD-LoRA (s42)"]
            value = {"DSv2": run[CROSS17.index("DSv2")], "CDFv3": run[CROSS17.index("CDFv3")],
                     "MEAN": run[CROSS17.index("Unseen")]}[col]
        else:
            value = got[S11_MAP[row]][COLS6[col]]
        assert agrees(printed, value), (row, col)


def test_sweeps_panel_matches_tableS12():
    text = run_script(SCRIPT, "sweeps")
    panels = {"Hinge margin m": "m=", "Small-transform penalty lambda_small": "lambda=",
              "GTL mixing constant c": "c=", "GSD weight eta_c=eta_h": "eta="}
    got = {}
    for line in text.splitlines():
        m = re.match(r"(m|lambda|c|eta)=(\S+)\s+epoch(\d+)\s+(\S+)\s+(\S+)$", line)
        if m:
            got[(m.group(1) + "=", m.group(2))] = (m.group(3), float(m.group(4)), float(m.group(5)))
    fx = load("tableS12")
    assert len(fx) == 30
    for (row, col), printed in fx.items():
        panel, value = [s.strip() for s in row.split("|")]
        prefix = panels[panel]
        value = value.replace("x10^", "e").replace("e-3", "e-3")
        key = next(k for k in got if k[0] == prefix and float(k[1]) == float(value))
        epoch, ffsh, m17 = got[key]
        if col == "Epoch":
            assert epoch == printed, (row, col)
        else:
            assert agrees(printed, {"FF-FSH": ffsh, "MEAN-17": m17}[col]), (row, col)


def _breakdown_sections(text):
    """{criterion: {label: [18 values]}} from the breakdown panel, per-run and mean rows alike."""
    out = {}
    for block in text.split("--- Criterion: ")[1:]:
        name, body = block.split(" ---", 1)
        section = {}
        for line in body.splitlines():
            if " mean (n=3)" in line:
                label = line.split(" mean (n=3)")[0] + " mean (n=3)"
                section[label] = [float(m) for m, _ in PM.findall(line)]
            else:
                parsed = rows(line)
                section.update({k: v for k, v in parsed.items() if len(v) == 18})
        out[name] = section
    return out


@pytest.mark.parametrize(
    "table_id,criterion",
    [("tableS13", "macro-best (default)"), ("tableS14", "DSv2-VAL only"),
     ("tableS15", "CDFv3-VAL only"), ("tableS16", "FF-FSH-VAL only")],
)
def test_breakdown_panel_matches_supp_tables(table_id, criterion):
    mod = _module()
    policy = {"macro-best (default)": "best_macro", "DSv2-VAL only": "best_dsv2",
              "CDFv3-VAL only": "best_cdfv3", "FF-FSH-VAL only": "best_ffsh"}[criterion]
    section = _breakdown_sections(run_script(SCRIPT, "breakdown"))[criterion]
    cfg_dir = {"LN": "orthoflow_gtl_ln", "LoRA": "orthoflow_gtl_lora", "GSD-LoRA": "orthoflow_gtl_gsd_lora"}
    fx = load(table_id)
    assert len(fx) == 216
    for (row, col), printed in fx.items():
        run = re.fullmatch(r"(\S+) \((s\d+), (epoch\d+)\)", row)
        if run:
            cfg, seed, epoch = run.groups()
            vals = section[f"{cfg} ({seed})"]
            found = mod.find_results(mod.MAIN_DIR / f"{cfg_dir[cfg]}_{seed}", policy)
            assert found.parent.name.startswith(epoch), (row, found.parent.name)
        else:
            vals = section[row]
        assert agrees(printed, vals[CROSS17.index(col)]), (row, col)
