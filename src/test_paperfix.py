from conftest import REPO, run_script
from paperfix import FIXTURES, agrees, load, rows


def test_agrees_formats():
    assert agrees("92.82", 92.8193) and agrees("+1.24", 1.2412) and agrees("0.015", 0.01507)
    assert not agrees("92.82", 92.8251)


def test_agrees_accepts_exact_ties():
    assert agrees("97.24", 97.235) and agrees("-1.27", -1.275) and agrees("97.23", 97.235)
    assert not agrees("92.82", 92.8251)


def test_rows_parses_fixed_width():
    t = "Configuration   n   DSv2  CDFv3  MEAN-17\n  + OPA          3  81.67  83.88    92.36\n"
    assert rows(t)["+ OPA"] == [3, 81.67, 83.88, 92.36]


def test_fixtures_cover_headline():
    assert load("table4")[("GSD-LoRA", "p")] == "0.015"
    assert load("table6")[("+ OPA + uniformity + GTL + GSD (GSD-LoRA)", "MEAN")] == "92.82"


def test_agrees_handles_signs_and_separators():
    assert agrees("−0.79", -0.7899) and not agrees("−0.79", 0.79)
    assert agrees("-0.0", -0.001) and agrees("1,655,290", 1655290.2)


def test_every_fixture_loads():
    ids = [f"table{i}" for i in range(1, 7)] + [f"tableS{i}" for i in range(1, 17) if i != 7]
    for table_id in ids:
        assert load(table_id), table_id
    assert not (FIXTURES / "tableS7.csv").exists()


def test_run_script_runs_from_repo_root(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("import os, matplotlib; print(os.getcwd()); print(matplotlib.get_backend())")
    out = run_script(str(script)).split()
    assert out[0] == str(REPO) and out[1].lower() == "agg"
