#!/usr/bin/env python3
"""
rebuttal_arcface_correlate.py — pair ArcFace identity embeddings with DINOv3
z_perp residuals, compute per-pair cosine similarities in both spaces, and
report the correlation split by manipulation type.

This is the analysis half of the orthogonality independent-verification
experiment for R2's circularity charge. The companion `rebuttal_arcface_embed.py`
extracts ArcFace embeddings; this script needs a separately-produced NPZ of
DINOv3 z_perp residuals on the same frames.

Expected DINOv3 z_perp NPZ schema (produced by your existing inference
pipeline; not shipped here — easiest path is to add an inference flag that
saves the residual computed in src.loss._orthoflow_gtl_core.OrthoflowGTLLoss._orthogonal_alignment
post-projection):
    ids:           (N,)        object array, must align with ArcFace ids
    z_perp:        (N, d)      float32 — unit-norm OPA residuals z_perp in
                                DINOv3 embedding space

Output
    A CSV listing each (real, fake) pair with arcface_cos and zperp_cos, plus
    a Markdown summary block with means split by manipulation type (face-swap
    vs reenactment) and the overall Pearson + Spearman correlation between
    the two similarity series.

Pair-matching rule
    Pairs are formed by joining on (video_id, frame_index, manipulation) from
    the metadata dict each NPZ carries. Real frames have label=="real",
    fake frames have label=="fake". One real-fake pair per (video, frame,
    manipulation) tuple.

Usage
    python rebuttal_arcface_correlate.py \
        --arcface  /path/to/arcface_embeddings.npz \
        --zperp    /path/to/zperp_embeddings.npz \
        --out-csv  /path/to/pair_similarities.csv \
        --out-md   /path/to/orthogonality_summary.md
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np


REENACTMENT_KEYWORDS = (
    # Speech-driven and reenactment-heavy manipulations across our suite
    "F2F", "NT",                    # FaceForensics++ reenactment
    "DSv", "FAVC", "PGF", "IDFv",   # AV / speech datasets
    "TF_", "LS_", "FR_",            # talking-head / lip-sync / reenactment prefixes in CDFv3, DSv*
)
SWAP_KEYWORDS = (
    "DF", "FS_", "FSH", "CDFv", "UADFV", "DFD", "DFDC", "KoDF", "FFIW", "DFDM",
)


def classify(manipulation: str) -> str:
    """Return 'reenactment', 'swap', or 'unknown' based on manipulation tag."""
    m = manipulation.upper()
    for kw in REENACTMENT_KEYWORDS:
        if kw.upper() in m:
            return "reenactment"
    for kw in SWAP_KEYWORDS:
        if kw.upper() in m:
            return "swap"
    return "unknown"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def pearson(xs: list[float], ys: list[float]) -> float:
    xs_a, ys_a = np.asarray(xs), np.asarray(ys)
    if len(xs_a) < 2:
        return float("nan")
    return float(np.corrcoef(xs_a, ys_a)[0, 1])


def spearman(xs: list[float], ys: list[float]) -> float:
    xs_a = np.asarray(xs).argsort().argsort().astype(np.float64)
    ys_a = np.asarray(ys).argsort().argsort().astype(np.float64)
    return pearson(xs_a.tolist(), ys_a.tolist())


def load_npz(path: str, key_emb: str) -> tuple[list[str], np.ndarray, list[dict], np.ndarray]:
    d = np.load(path, allow_pickle=True)
    ids = list(d["ids"].tolist())
    embs = d[key_emb]
    metas = [json.loads(m) for m in d["metadata"].tolist()]
    valid = d["valid"] if "valid" in d.files else np.ones(len(ids), dtype=bool)
    return ids, embs, metas, valid


def pair_key(meta: dict) -> tuple:
    """Build the join key used to match real frames to fake frames."""
    return (meta.get("video_id"), meta.get("frame_index"), meta.get("manipulation"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arcface", required=True, help="NPZ of ArcFace embeddings")
    p.add_argument("--zperp", required=True, help="NPZ of DINOv3 z_perp residuals")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args()

    arc_ids, arc_emb, arc_meta, arc_valid = load_npz(args.arcface, "embeddings")
    z_ids, z_emb, z_meta, z_valid = load_npz(args.zperp, "z_perp")

    # Index both NPZs by frame_id
    arc_by_id = {fid: (arc_emb[i], arc_meta[i], arc_valid[i]) for i, fid in enumerate(arc_ids)}
    z_by_id = {fid: (z_emb[i], z_meta[i], z_valid[i]) for i, fid in enumerate(z_ids)}

    # For each real frame, find its matched fake frame (same video_id, frame_index)
    reals: dict[tuple, str] = {}  # key (video_id, frame_index, manipulation) -> real_frame_id
    fakes: dict[tuple, str] = {}
    for fid, (_, meta, ok) in arc_by_id.items():
        if not ok or fid not in z_by_id:
            continue
        z_ok = z_by_id[fid][2]
        if not z_ok:
            continue
        key = pair_key(meta)
        label = meta.get("label", "")
        if label == "real":
            reals[key] = fid
        elif label == "fake":
            fakes[key] = fid

    rows = []
    for key, fid_r in reals.items():
        fid_f = fakes.get(key)
        if fid_f is None:
            continue
        arc_r, _, _ = arc_by_id[fid_r]
        arc_f, _, _ = arc_by_id[fid_f]
        z_r, _, _ = z_by_id[fid_r]
        z_f, _, _ = z_by_id[fid_f]
        arc_cos = cosine(arc_r, arc_f)
        z_cos = cosine(z_r, z_f)
        rows.append({
            "video_id": key[0], "frame_index": key[1], "manipulation": key[2],
            "type": classify(key[2] or ""),
            "real_frame_id": fid_r, "fake_frame_id": fid_f,
            "arcface_cos": arc_cos, "zperp_cos": z_cos,
        })

    if not rows:
        print("no matched pairs — check that arcface and zperp NPZs share frame_ids "
              "and that each pair has both real and fake entries", file=sys.stderr)
        return 1

    # Write per-pair CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate by manipulation type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    overall = rows
    overall_arc = [r["arcface_cos"] for r in overall]
    overall_z = [r["zperp_cos"] for r in overall]
    pearson_overall = pearson(overall_arc, overall_z)
    spearman_overall = spearman(overall_arc, overall_z)

    md_lines = []
    md_lines.append("# Orthogonality independent verification (ArcFace vs DINOv3 z_perp)\n")
    md_lines.append(f"Pairs evaluated: **{len(rows)}**\n")
    md_lines.append(f"Overall Pearson r(arcface_cos, zperp_cos) = **{pearson_overall:+.4f}**")
    md_lines.append(f"Overall Spearman ρ(arcface_cos, zperp_cos) = **{spearman_overall:+.4f}**\n")
    md_lines.append("## Per-manipulation-type means\n")
    md_lines.append("| type | n | arcface_cos (μ ± σ) | zperp_cos (μ ± σ) | Pearson r |")
    md_lines.append("|------|---|---------------------|-------------------|-----------|")
    for tname in ("swap", "reenactment", "unknown"):
        rs = by_type.get(tname, [])
        if not rs:
            continue
        a = np.asarray([r["arcface_cos"] for r in rs])
        z = np.asarray([r["zperp_cos"] for r in rs])
        a = a[~np.isnan(a)]; z = z[~np.isnan(z)]
        if len(a) < 1: continue
        r_p = pearson(a.tolist(), z.tolist()) if len(a) > 1 else float("nan")
        md_lines.append(
            f"| {tname} | {len(rs)} | "
            f"{a.mean():+.3f} ± {a.std():.3f} | "
            f"{z.mean():+.3f} ± {z.std():.3f} | "
            f"{r_p:+.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Interpretation key\n")
    md_lines.append("- **Swap pairs**: ArcFace should be LOW (identity changed); z_perp prediction is method-dependent.")
    md_lines.append("- **Reenactment pairs**: ArcFace should be HIGH (identity preserved); z_perp should also be HIGH if our claim that z_perp captures identity in this case holds.")
    md_lines.append("- A positive Pearson correlation overall (or within reenactment) breaks R2's circularity charge: z_perp tracks identity as measured by a model trained on a wholly different objective and embedding space.")

    with open(args.out_md, "w") as f:
        f.write("\n".join(md_lines))

    print(f"wrote {args.out_csv} ({len(rows)} pairs) and {args.out_md}")
    print(f"  Overall Pearson r = {pearson_overall:+.4f}, Spearman ρ = {spearman_overall:+.4f}")
    for tname, rs in by_type.items():
        print(f"  {tname:12} n={len(rs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
