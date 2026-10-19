"""Checks on the SLERP rerun evidence: per-run MEAN-17 values and the matched on-off deltas."""
from pathlib import Path

import pandas as pd
import pytest
from paperfix import agrees

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm,on,off,delta", [("ln", "91.64", "91.51", "+0.13"), ("lora", "92.18", "92.64", "-0.46"), ("gsd_lora", "92.40", "92.89", "-0.49")])
def test_rerun_means(arm, on, off, delta):
    d = pd.read_csv(ROOT / "results/slerp_rerun/runs.csv")
    a = d[d.arm == arm]; m_on = a[a.slerp == "on"].mean17.mean(); m_off = a[a.slerp == "off"].mean17.mean()
    assert agrees(on, m_on) and agrees(off, m_off) and agrees(delta, m_on - m_off) and len(a) == 6
