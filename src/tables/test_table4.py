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


def test_gsd_over_lora():
    """The paper's ablation discussion: GSD adds +0.20 over plain LoRA, p 0.40 (stated in the text, not in a table)."""
    r = wilcoxon_vs(load_cm(), "GSD-LoRA", reference="lora")
    assert agrees("+0.20", r["delta"]) and agrees("0.40", r["p"])


def test_gsd_over_lora_cli():
    import os
    import subprocess
    import sys

    env = dict(os.environ, MPLBACKEND="Agg")
    done = subprocess.run([sys.executable, "src/tables/table4_wilcoxon.py", "--reference", "lora"], cwd=REPO,
                          env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert any(line.startswith("GSD-LoRA 92.82 +0.20 0.40 ") for line in done.stdout.splitlines()), done.stdout


def _cli(*args):
    import os
    import subprocess
    import sys

    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, "src/tables/table4_wilcoxon.py", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, check=False)


def test_cli_leaves_out_the_variant_that_is_the_reference():
    done = _cli("--reference", "lora")
    assert done.returncode == 0, done.stderr
    assert [line.split()[0] for line in done.stdout.strip().splitlines()[1:]] == ["LN", "GSD-LoRA"]


def test_cli_accepts_only_known_references():
    done = _cli("--reference", "GenD-DINOv3")
    assert done.returncode == 2
    assert "invalid choice" in done.stderr
    for ref in ("GenD-DINO", "GenD_DINOv3_L", "ForensicAdapter_v1", "ln", "lora", "gsd_lora"):
        assert _cli("--reference", ref).returncode == 0, ref
