import pytest

from conftest import REPO, run_script

SUPP = ["figS1_heatmap", "figS2_per_dataset", "figS3_summary", "figS4_delta_gsd_vs_gend"]
MAGIC = {"pdf": b"%PDF", "png": b"\x89PNG"}


def _assert_valid(path):
    suffix = path.suffix.lstrip(".")
    assert path.exists(), f"{path.name} was not written"
    assert path.stat().st_size > 10_000
    assert path.read_bytes()[: len(MAGIC[suffix])] == MAGIC[suffix]


def test_fig2_written(tmp_path):
    run_script("src/figures/make_figure2.py", str(tmp_path))
    for suffix in ("pdf", "png"):
        _assert_valid(tmp_path / f"fig2_val_curves.{suffix}")


@pytest.fixture(scope="module")
def comparison_run():
    """Run the comparison script once and record which targets it freshly wrote."""
    targets = [REPO / "figures" / f"{stem}.{suffix}" for stem in SUPP for suffix in ("pdf", "png")]
    before = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in targets}
    run_script("src/figures/build_comparison_figures.py")
    return before


@pytest.mark.parametrize("stem", SUPP)
def test_supplementary_comparison_figures_written(stem, comparison_run):
    for suffix in ("pdf", "png"):
        path = REPO / "figures" / f"{stem}.{suffix}"
        _assert_valid(path)
        assert path.stat().st_mtime_ns != comparison_run[path], f"{path.name} was not rewritten"
