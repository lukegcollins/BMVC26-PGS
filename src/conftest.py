import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_script(rel_path: str, *args: str) -> str:
    """Run a repository script with the current interpreter from the repository root and return its stdout."""
    env = dict(os.environ, MPLBACKEND="Agg", SOURCE_DATE_EPOCH="0")
    done = subprocess.run(
        [sys.executable, rel_path, *args], cwd=REPO, env=env, check=False, capture_output=True, text=True
    )
    assert done.returncode == 0, f"{rel_path} exited {done.returncode}\n{done.stderr}"
    return done.stdout
