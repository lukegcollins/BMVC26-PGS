import pytest

from conftest import REPO, run_script

FIG2 = ["fig2_val_curves"]
SUPP = ["figS1_heatmap", "figS2_per_dataset", "figS3_summary", "figS4_delta_gsd_vs_gend"]


def _check(stem: str) -> None:
    for suffix in ("pdf", "png"):
        path = REPO / "figures" / f"{stem}.{suffix}"
        assert path.exists(), f"{path.name} was not written"
        assert path.stat().st_size > 10_000
    assert (REPO / "figures" / f"{stem}.pdf").read_bytes()[:4] == b"%PDF"


def test_fig2_written():
    run_script("src/figures/make_figure2.py")
    for stem in FIG2:
        _check(stem)


@pytest.mark.parametrize("stem", SUPP)
def test_supplementary_comparison_figures_written(stem):
    run_script("src/figures/build_comparison_figures.py")
    _check(stem)
