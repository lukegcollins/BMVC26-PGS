"""Table 4: exact two-sided Wilcoxon signed-rank test of each Ours variant against GenD-DINO.

Reads `results/baselines/comparison_master.csv`. The test runs over the 17 cross-dataset columns that make up
MEAN-17. Delta is MEAN-17(ours) minus MEAN-17(reference); wins counts the datasets where ours scores higher.
Run from any directory; prints one row per variant: Variant MEAN-17 Delta p Wins.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "results" / "baselines" / "comparison_master.csv"

# the 17 cross-dataset columns of MEAN-17, in the order the emitters use
XD17 = ["FF-FSH-C23", "CDFv1", "CDFv2", "CDFv3", "DeFo", "DFD-C23", "DSv1", "DSv2",
        "DFDCP", "DFDC", "DFDM-C23", "FAVC", "FFIW10K", "IDFv1", "KoDF", "PGF", "UADFV"]
VARIANTS = {"LN": "ln", "LoRA": "lora", "GSD-LoRA": "gsd_lora"}
REFERENCE = "GenD-DINO"
REFERENCE_ROW = "GenD_DINOv3_L"


def wilcoxon_vs(cm: pd.DataFrame, variant: str, reference: str = REFERENCE) -> dict:
    """Return MEAN-17, delta, exact two-sided Wilcoxon p, wins and n for one Ours variant against the reference."""
    ref_row = REFERENCE_ROW if reference == REFERENCE else reference
    ours_row = VARIANTS[variant]
    ours = cm.loc[ours_row, XD17].astype(float)
    ref = cm.loc[ref_row, XD17].astype(float)
    result = wilcoxon(ours.to_numpy(), ref.to_numpy(), alternative="two-sided", method="exact")
    return {
        "mean17": float(cm.loc[ours_row, "MEAN_17"]),
        "delta": float(cm.loc[ours_row, "MEAN_17"]) - float(cm.loc[ref_row, "MEAN_17"]),
        "p": float(result.pvalue),
        "wins": int((ours.to_numpy() > ref.to_numpy()).sum()),
        "n": len(XD17),
    }


def _format_p(p: float) -> str:
    """Three decimals below 0.1, as the paper prints small p-values; two otherwise."""
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", default=REFERENCE,
                        help="reference method: GenD-DINO (default) or a comparison_master.csv row id such as lora")
    args = parser.parse_args(argv)
    cm = pd.read_csv(MASTER, index_col=0)
    print("Variant MEAN-17 Delta p Wins")
    for variant in VARIANTS:
        r = wilcoxon_vs(cm, variant, reference=args.reference)
        print(f"{variant} {r['mean17']:.2f} {r['delta']:+.2f} {_format_p(r['p'])} {r['wins']}/{r['n']}")


if __name__ == "__main__":
    main()
