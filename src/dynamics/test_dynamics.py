import re

from conftest import REPO, run_script
from paperfix import agrees, load

COLUMNS = {
    "Peak epoch": 0,
    "Peak val-AUC (%)": 1,
    "Post-peak slope (pp/epoch)": 2,
    "AUC-under-curve (%)": 3,
}


def _parse(stdout: str) -> dict[str, list[float]]:
    parsed = {}
    for line in stdout.splitlines():
        match = re.match(r"^(LN|LoRA|GSD-LoRA)\s+(.*)$", line)
        if match:
            parsed[match.group(1)] = [float(t) for t in match.group(2).split()]
    return parsed


def test_table5_matches_paper():
    parsed = _parse(run_script("src/dynamics/table5.py"))
    printed = load("table5")
    assert len(printed) == 12
    for (row, col), value in printed.items():
        assert row in parsed, f"missing row {row}"
        regenerated = parsed[row][COLUMNS[col]]
        assert agrees(value, regenerated), f"{row} / {col}: printed {value}, regenerated {regenerated}"


def test_figS6_written():
    targets = [REPO / "figures" / f"figS6_training_dynamics.{suffix}" for suffix in ("pdf", "png")]
    before = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in targets}
    run_script("src/dynamics/build_training_dynamics_figure.py")
    for path in targets:
        assert path.exists() and path.stat().st_size > 10_000
        assert path.stat().st_mtime_ns != before[path], f"{path.name} was not rewritten"
        assert path.read_bytes()[:4] == (b"%PDF" if path.suffix == ".pdf" else b"\x89PNG")


def test_slope_needs_two_points_from_the_peak():
    import math

    import pandas as pd

    from dynamics.table5 import per_run

    def curve(run, aucs):
        return pd.DataFrame({"run": run, "variant": "ln", "seed": 42, "epoch": range(len(aucs)), "criterion": "macro",
                             "val_loss": 0.5, "val_video_auc": aucs})

    runs = per_run(pd.concat([curve("peak_last", [0.80, 0.82, 0.85]), curve("peak_middle", [0.80, 0.85, 0.84])]))
    slopes = dict(zip(["peak_last", "peak_middle"], runs["slope"]))
    assert math.isnan(slopes["peak_last"])
    assert agrees("-1.00", slopes["peak_middle"])
