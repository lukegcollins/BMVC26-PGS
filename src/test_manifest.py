"""MANIFEST.md covers every paper element, and every command in it runs from the repository root."""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.md"

# every \label{tab:...} and \label{fig:...} of the corrected paper (main text and supplement)
EXPECTED = {
    "fig:pipeline", "tab:impl_details", "tab:indomain_results", "tab:cross_results", "tab:headline_deltas",
    "tab:training_stability", "fig:val_curves", "fig:lora_dynamics", "fig:training_dynamics",
    "fig:advantage_and_dynamics", "tab:ablation",
    "tab:supp_dataset_stats", "tab:supp_pub_vs_reeval_cross", "tab:supp_pub_vs_reeval_indomain",
    "tab:supp_orig_published_cross", "tab:supp_mean13_subset", "fig:supp_heatmap", "fig:supp_perdataset",
    "fig:supp_summary", "fig:delta_gsd_vs_gend", "tab:supp_checkpoint_policy", "tab:k_sweep", "fig:centroid_map",
    "tab:supp_arcface_pertype", "tab:supp_arcface_idsep", "tab:supp_loss_weights", "tab:supp_ablation_sensitivity",
    "tab:supp_hyperparam_sweeps", "fig:supp_box_summary", "tab:supp_full_breakdown_macro",
    "tab:supp_full_breakdown_dsv2", "tab:supp_full_breakdown_cdfv3", "tab:supp_full_breakdown_ffsh",
}

NEEDS_ASSETS = "regenerates with release assets"
ASSET_FILES = ("arcface_embeddings.npz", "zperp_embeddings.npz")
COMMAND = re.compile(r"`uv run python ([^`]+)`")


def manifest_rows(path: Path) -> list[list[str]]:
    """Cells of every table row that has five columns (Label | What it shows | Script | Artefact | Status)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and not set(cells[0]) <= {"-", ":"} and cells[0] != "Label":
            rows.append(cells)
    return rows


def manifest_labels(path: Path) -> set[str]:
    labels = set()
    for cells in manifest_rows(path):
        labels.update(re.findall(r"\b(?:tab|fig):[A-Za-z0-9_]+", cells[0]))
    return labels


def manifest_commands(path: Path) -> list[tuple[str, str]]:
    """(command arguments, status) for every `uv run python ...` in the Script column."""
    found = []
    for cells in manifest_rows(path):
        for args in COMMAND.findall(cells[2]):
            if (args, cells[4]) not in found:
                found.append((args, cells[4]))
    return found


def _tracked_state() -> str | None:
    done = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else None


def test_manifest_covers_every_paper_element():
    labels = manifest_labels(MANIFEST)
    assert EXPECTED <= labels, sorted(EXPECTED - labels)


def test_manifest_has_commands():
    assert len(manifest_commands(MANIFEST)) >= 10


def test_manifest_commands_run_from_root():
    """Run each command from the repository root with the figure-test environment; figures stay byte-identical."""
    env = dict(os.environ, MPLBACKEND="Agg", SOURCE_DATE_EPOCH="0")
    have_assets = all((ROOT / "data" / "arcface" / name).exists() for name in ASSET_FILES)
    before = _tracked_state()
    ran = 0
    skipped = []
    for args, status in manifest_commands(MANIFEST):
        if status.startswith(NEEDS_ASSETS) and not have_assets:
            skipped.append(args)
            continue
        done = subprocess.run(
            [sys.executable, *shlex.split(args)], cwd=ROOT, env=env, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, f"{args}\n{done.stderr}"
        ran += 1
    assert ran > 0
    if before is not None:
        assert _tracked_state() == before, "running the MANIFEST commands changed tracked files"
    if skipped:
        pytest.skip(f"ran {ran} commands; skipped {len(skipped)} that need data/arcface/*.npz (release assets)")
