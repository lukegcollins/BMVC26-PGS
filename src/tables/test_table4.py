"""Table 4: exact two-sided Wilcoxon signed-rank tests of each Ours variant against GenD-DINO over the 17 datasets."""

import pandas as pd
import pytest

from conftest import REPO
from paperfix import agrees, load
from tables.table4_wilcoxon import wilcoxon_vs

MASTER = REPO / "results" / "baselines" / "comparison_master.csv"


def load_cm() -> pd.DataFrame:
    return pd.read_csv(MASTER, index_col=0)


@pytest.mark.parametrize("v,delta,p,wins", [("LN", "+0.39", "0.43", 11), ("LoRA", "+1.04", "0.020", 13),
                                            ("GSD-LoRA", "+1.24", "0.015", 13)])
def test_table4(v, delta, p, wins):
    r = wilcoxon_vs(load_cm(), v)
    assert agrees(delta, r["delta"]) and agrees(p, r["p"]) and r["wins"] == wins and r["n"] == 17


def test_table4_matches_fixture():
    fixture = load("table4")
    for variant in ("LN", "LoRA", "GSD-LoRA"):
        r = wilcoxon_vs(load_cm(), variant)
        assert agrees(fixture[(variant, "MEAN-17 (%)")], r["mean17"]), variant
        assert agrees(fixture[(variant, "Δ")], r["delta"]), variant
        assert agrees(fixture[(variant, "p")], r["p"]), variant
        assert fixture[(variant, "Wins")] == f"{r['wins']}/{r['n']}", variant


def test_table4_cli_prints_one_row_per_variant():
    import os
    import subprocess
    import sys

    env = dict(os.environ, MPLBACKEND="Agg")
    done = subprocess.run([sys.executable, "src/tables/table4_wilcoxon.py"], cwd=REPO, env=env,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.strip().splitlines()
    assert lines[0].split() == ["Variant", "MEAN-17", "Delta", "p", "Wins"]
    rows = [line.split() for line in lines[1:]]
    assert [r[0] for r in rows] == ["LN", "LoRA", "GSD-LoRA"]
    assert all(len(r) == 5 for r in rows)
    assert all(r[2][0] in "+-" for r in rows)
