"""Tables S8 and S9 (across-pair ArcFace identity check) regenerate from the released embedding arrays."""

import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conftest import run_script
from paperfix import agrees, load

ROOT = SRC.parent
NPZ = ROOT / "data/arcface"
pytestmark = pytest.mark.skipif(
    not all((NPZ / f).exists() for f in ("arcface_embeddings.npz", "zperp_embeddings.npz")),
    reason="ArcFace arrays are release assets: download them into data/arcface/ (see CHECKPOINTS.md)",
)

# printed value is the mean over the shipped 10,000-row sample; the full population gives 0.0540
KNOWN_SOURCE = {("all combinations", "mean cos(z_perp)"): "sample"}

NUM = r"([+-]?[\d,]+(?:\.\d+)?)"


SAMPLE = []


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    out = tmp_path_factory.mktemp("arcface") / "summary.md"
    sample = out.parent / "sample.csv"
    SAMPLE.append(sample)
    run_script(
        "src/arcface/correlate_across_pairs.py",
        "--arcface", str(NPZ / "arcface_embeddings.npz"),
        "--zperp", str(NPZ / "zperp_embeddings.npz"),
        "--out-md", str(out),
        "--out-csv-sample", str(sample),
    )
    return out.read_text(encoding="utf-8")


def _num(token):
    return float(token.replace(",", ""))


def _sample_mean(col):
    import csv

    with open(ROOT / "results/arcface/pair_pair_sample.csv", newline="") as h:
        vals = [float(r[col]) for r in csv.DictReader(h)]
    return sum(vals) / len(vals)


def _check(cells):
    bad = [(r, c, p, v) for (r, c), (p, v) in cells.items() if not agrees(p, v)]
    assert not bad, f"mismatches (row, col, printed, regenerated): {bad}"


def test_tableS8_matches_paper(summary):
    fx = load("tableS8")
    cells = {}
    for label, key in (("swap x swap", "swap x swap"), ("swap x reenact", "swap x reenact"),
                       ("reenact x reenact", "reenact x reenact")):
        m = re.search(rf"\| {label} \| {NUM} \| {NUM} \| {NUM} \| {NUM} \|", summary)
        assert m, label
        for col, g in zip(("n", "mean cos(arc)", "mean cos(z_perp)", "Pearson r"), m.groups()):
            cells[(key, col)] = (fx[(key, col)], _num(g))
    n_all = re.search(r"pair-pair combos: \*\*([\d,]+)\*\*", summary)
    r_all = re.search(r"Pearson r\(cos_z, cos_arc\) = \*\*" + NUM + r"\*\*", summary)
    assert n_all and r_all
    cells[("all combinations", "n")] = (fx[("all combinations", "n")], _num(n_all.group(1)))
    cells[("all combinations", "Pearson r")] = (fx[("all combinations", "Pearson r")], _num(r_all.group(1)))
    # The overall means are the n-weighted means of the three combos.
    for col, idx in (("mean cos(arc)", 1), ("mean cos(z_perp)", 2)):
        tot = 0.0
        for label in ("swap x swap", "swap x reenact", "reenact x reenact"):
            m = re.search(rf"\| {label} \| {NUM} \| {NUM} \| {NUM} \| {NUM} \|", summary)
            tot += _num(m.group(1)) * _num(m.group(idx + 1))
        full = tot / _num(n_all.group(1))
        if KNOWN_SOURCE.get(("all combinations", col)) == "sample":
            assert agrees("+0.0540", full), full
            cells[("all combinations", col)] = (fx[("all combinations", col)], _sample_mean("cos_z"))
        else:
            cells[("all combinations", col)] = (fx[("all combinations", col)], full)
    _check(cells)


def test_tableS9_matches_paper(summary):
    fx = load("tableS9")
    cells = {}
    parts = {}
    for label in ("same target identity", "different target identity"):
        m = re.search(rf"\| {label} \| {NUM} \| {NUM} \| {NUM} \|", summary)
        assert m, label
        n, a, z = (_num(g) for g in m.groups())
        parts[label] = (a, z)
        cells[(label, "n")] = (fx[(label, "n")], n)
        cells[(label, "mean cos(arc)")] = (fx[(label, "mean cos(arc)")], a)
        cells[(label, "mean cos(z_perp)")] = (fx[(label, "mean cos(z_perp)")], z)
    sep = "separation Δ (same−different)"
    s = parts["same target identity"]
    d = parts["different target identity"]
    cells[(sep, "mean cos(arc)")] = (fx[(sep, "mean cos(arc)")], s[0] - d[0])
    cells[(sep, "mean cos(z_perp)")] = (fx[(sep, "mean cos(z_perp)")], s[1] - d[1])
    _check(cells)


def test_matches_shipped_summary(summary):
    shipped = (ROOT / "results/arcface/orthogonality_summary.md").read_text(encoding="utf-8")
    assert summary.strip() == shipped.strip()


def test_sample_matches_shipped(summary):
    # Same seed and arrays: the sampled rows are the shipped pair_pair_sample.csv (cosines to float32 rounding).
    import csv

    with open(SAMPLE[0], newline="") as h:
        got = list(csv.DictReader(h))
    with open(ROOT / "results/arcface/pair_pair_sample.csv", newline="") as h:
        want = list(csv.DictReader(h))
    assert len(got) == len(want)
    for g, w in zip(got, want):
        for k in ("i", "j", "target_i", "target_j", "type_combo", "same_identity"):
            assert g[k] == w[k]
        for k in ("cos_arc", "cos_z"):
            assert abs(float(g[k]) - float(w[k])) <= 2e-6
