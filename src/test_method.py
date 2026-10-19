"""Checks on the frozen method code and the SLERP restoration patch."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AS_RUN = "2813161f"
_WB = os.environ.get("PGS_WORKBENCH")
WB = Path(_WB) if _WB else None
HAVE_WB = WB is not None and WB.exists()


def git_show(repo: Path, commit: str, rel: Path) -> bytes:
    out = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{rel.as_posix()}"], capture_output=True, check=True
    )
    return out.stdout


def test_patch_applies():
    r = subprocess.run(
        ["git", "apply", "--check", "patches/slerp-restore.patch"], cwd=ROOT, capture_output=True, check=False
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not HAVE_WB, reason="byte check needs the private Workbench repository")
def test_method_files_are_as_run():
    files = sorted((ROOT / "method/src").rglob("*.py"))
    assert files, "method/src holds no Python files"
    for f in files:
        rel = f.relative_to(ROOT / "method")
        assert f.read_bytes() == git_show(WB, AS_RUN, rel), rel
