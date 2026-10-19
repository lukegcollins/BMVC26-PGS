"""Checks on the copied evidence layout: run logs, configs and repository size."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = [f"orthoflow_gtl_{v}_s{s}" for v in ("ln", "lora", "gsd_lora") for s in (42, 137, 1729)]


def yaml_flag(path: Path, key: str):
    """Return the boolean value of the first `key: true|false` line (any indent), or None if absent."""
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(true|false)\b", re.MULTILINE)
    match = pattern.search(path.read_text())
    if match is None:
        return None
    return match.group(1) == "true"


def is_git_checkout(root: Path) -> bool:
    """True when ``root`` is the top level of a git working tree (not, say, an unpacked "Download ZIP")."""
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True,
                             check=False)
    except FileNotFoundError:
        return False
    return out.returncode == 0 and Path(out.stdout.strip()).resolve() == root.resolve()


def git_ls_files(root: Path):
    out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def test_headline_logs_present():
    for r in RUNS:
        assert (ROOT / "results/logs/main" / r / "config.yaml").is_file()


def test_configs_record_slerp_off():
    ys = list((ROOT / "configs").rglob("*.yaml"))
    assert len(ys) == 46
    for y in ys:
        assert yaml_flag(y, "use_paired_slerp_aug") is False


def test_no_bulk_in_tree():
    if not is_git_checkout(ROOT):
        pytest.skip("not a git checkout (for example a Download ZIP), so there is no list of tracked files to check")
    tracked = git_ls_files(ROOT)
    assert not [f for f in tracked if f.endswith((".npz", ".safetensors", ".ckpt"))]
    assert not [f for f in tracked if (ROOT / f).stat().st_size > 5 * 1024 * 1024]


def test_is_git_checkout(tmp_path):
    assert not is_git_checkout(tmp_path)
    assert is_git_checkout(ROOT) == (ROOT / ".git").exists()
