"""Printed-number fixtures for the paper tables, and the helpers that compare regenerated values with them."""

import csv
import re
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "paper"

_MINUS = "−"
_NUMERIC_TOKEN = re.compile(r"(?<!\S)[+−-]?\d[\d,]*(?:\.\d+)?(?!\S)")


def load(table_id: str) -> dict[tuple[str, str], str]:
    """Return {(row, col): number as printed} for one paper table, for example ("GSD-LoRA", "MEAN") -> "92.82"."""
    with open(FIXTURES / f"{table_id}.csv", newline="", encoding="utf-8") as handle:
        return {(r["row"], r["col"]): r["value"] for r in csv.DictReader(handle)}


def _to_float(token: str) -> float:
    return float(token.replace(_MINUS, "-").replace(",", "").lstrip("+"))


def _fixed(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    return text[1:] if text.startswith("-") and not text.strip("-0.") else text


def agrees(printed: str, value: float) -> bool:
    """True when `value`, formatted to the printed number's decimals, equals the printed number."""
    text = printed.strip().replace(",", "")
    decimals = len(text.split(".")[1]) if "." in text else 0
    return _fixed(value, decimals) == _fixed(_to_float(text), decimals)


def rows(text: str) -> dict[str, list[float]]:
    """Parse a script's stdout table: label = text before the first numeric token, values = every float after it."""
    parsed: dict[str, list[float]] = {}
    for line in text.splitlines():
        first = _NUMERIC_TOKEN.search(line)
        if first is None:
            continue
        label = line[: first.start()].strip()
        parsed[label] = [_to_float(t) for t in _NUMERIC_TOKEN.findall(line[first.start() :])]
    return parsed
