"""Each released model is the checkpoint the paper evaluated: the one named in its run's best-macro ``test.log``."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SIDECARS = sorted((REPO / "checkpoints").glob("*.json"))
TESTING = re.compile(r"Testing checkpoint: (?:.*/)?([^/\s]+/checkpoints/[^/\s]+\.ckpt)\s*$")
RESULTS_DIR = re.compile(r"^epoch(\d+)_best_macro$")


def _reproduce_tables():
    path = REPO / "src" / "tables" / "reproduce_tables.py"
    spec = importlib.util.spec_from_file_location("reproduce_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RT = _reproduce_tables()


def test_there_are_nine_sidecars():
    assert [p.stem for p in SIDECARS] == sorted(
        f"orthoflow_gtl_{v}_s{s}" for v in ("ln", "lora", "gsd_lora") for s in (42, 137, 1729)
    )


@pytest.mark.parametrize("sidecar", SIDECARS, ids=lambda p: p.stem)
def test_sidecar_is_the_checkpoint_the_paper_evaluated(sidecar):
    side = json.loads(sidecar.read_text(encoding="utf-8"))
    run = side["run"]
    results = RT.find_results(RT.MAIN_DIR / run, "best_macro")
    assert results is not None, f"{run}: no best_macro test results"

    m = RESULTS_DIR.match(results.parent.name)
    assert m, results.parent.name
    assert side["epoch"] == int(m.group(1))

    first = (results.parent / "test.log").read_text(encoding="utf-8").splitlines()[0]
    tested = TESTING.search(first)
    assert tested, first
    evaluated = side.get("evaluated_as", {}).get("source_file", side["source_file"])
    assert tested.group(1) == evaluated


def test_only_gsd_lora_s1729_was_evaluated_under_another_name():
    aliased = {}
    for p in SIDECARS:
        side = json.loads(p.read_text(encoding="utf-8"))
        if "evaluated_as" in side:
            aliased[side["run"]] = side
    assert list(aliased) == ["orthoflow_gtl_gsd_lora_s1729"]
    side = aliased["orthoflow_gtl_gsd_lora_s1729"]
    assert side["source_file"] == "orthoflow_gtl_gsd_lora_s1729/checkpoints/epoch=07-video_auc=0.8651.ckpt"
    assert side["evaluated_as"]["source_file"] == "orthoflow_gtl_gsd_lora_s1729/checkpoints/best_CDFv3-VAL.ckpt"
    assert side["evaluated_as"]["state_dict_identical"] is True
    assert side["evaluated_as"]["state_dict_tensors"] == 724
