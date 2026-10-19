#!/usr/bin/env python3
"""correlate_across_pairs.py — Option A correlation: across-pair z_perp.

The within-pair comparison `cos(z_perp_real, z_perp_fake)` is degenerate by
construction (the OPA orthogonalization makes the two equal). The meaningful
test is whether the manipulation-orthogonal DINOv3 subspace preserves identity
structure that an independent face-recognition model agrees with.

For each FF++ pair P_i (target identity + frame + manipulation method) we have:
  - z_perp_Pi: one OPA orthogonal residual per pair (real == fake by construction)
  - arc_real_Pi: ArcFace embedding of the real reference frame
  - type_Pi: swap (DF, FS) or reenactment (F2F, NT)
  - target_id_Pi: the target identity (== FF++ source video id)

For each unique (i, j) pair-pair combo (i < j) we compute:
  - cos_z = cos(z_perp_Pi, z_perp_Pj)
  - cos_a = cos(arc_real_Pi, arc_real_Pj)
  - type_combo: swap×swap | swap×reenact | reenact×reenact
  - same_identity: target_id_Pi == target_id_Pj

Reports Pearson/Spearman of (cos_z, cos_a) overall + per type-combo + the
mean cosines partitioned by same-identity vs different-identity. The latter
is the headline number: if z_perp preserves identity, same-identity pair-pair
combos should have substantially higher cos_z than different-identity combos.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


REENACTMENT_KEYWORDS = (
    "F2F", "NT", "FACE2FACE", "NEURALTEXTURES",
)
SWAP_KEYWORDS = (
    "DF", "FS_", "DEEPFAKES", "FACESWAP",
)


def classify(manipulation: str) -> str:
    m = (manipulation or "").upper()
    for kw in REENACTMENT_KEYWORDS:
        if kw.upper() in m:
            return "reenactment"
    for kw in SWAP_KEYWORDS:
        if kw.upper() in m:
            return "swap"
    return "unknown"


def pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    xs_r = xs.argsort().argsort().astype(np.float64)
    ys_r = ys.argsort().argsort().astype(np.float64)
    return pearson(xs_r, ys_r)


def load_npz(path: str, emb_key: str):
    d = np.load(path, allow_pickle=True)
    ids = list(d["ids"].tolist())
    embs = d[emb_key]
    metas = [json.loads(m) for m in d["metadata"].tolist()]
    valid = d["valid"] if "valid" in d.files else np.ones(len(ids), dtype=bool)
    return ids, embs, metas, valid


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arcface", required=True)
    p.add_argument("--zperp", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-csv-sample", default=None,
                   help="Optional: save a random 10k-row sample of pair-pair combos for inspection")
    p.add_argument("--sample-rows", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arc_ids, arc_emb, arc_meta, arc_valid = load_npz(args.arcface, "embeddings")
    z_ids, z_emb, z_meta, z_valid = load_npz(args.zperp, "z_perp")

    arc_by_id = {fid: (arc_emb[i], arc_meta[i], bool(arc_valid[i])) for i, fid in enumerate(arc_ids)}
    z_by_id = {fid: (z_emb[i], z_meta[i], bool(z_valid[i])) for i, fid in enumerate(z_ids)}

    # Build per-pair records by joining real entries (one record per pair)
    pairs = []
    for fid, (_, meta, ok) in arc_by_id.items():
        if not ok:
            continue
        if meta.get("label") != "real":
            continue
        if fid not in z_by_id or not z_by_id[fid][2]:
            continue
        pairs.append({
            "frame_id": fid,
            "target_id": meta.get("video_id"),
            "frame_index": meta.get("frame_index"),
            "manipulation": meta.get("manipulation"),
            "type": classify(meta.get("manipulation") or ""),
            "arc": arc_by_id[fid][0],
            "z": z_by_id[fid][0],
        })

    n = len(pairs)
    print(f"loaded {n} pairs")

    arc_mat = np.stack([p["arc"] for p in pairs]).astype(np.float32)
    z_mat = np.stack([p["z"] for p in pairs]).astype(np.float32)
    # both are already L2-normalized so cosine = dot product
    arc_norms = np.linalg.norm(arc_mat, axis=1, keepdims=True)
    z_norms = np.linalg.norm(z_mat, axis=1, keepdims=True)
    arc_mat = arc_mat / (arc_norms + 1e-8)
    z_mat = z_mat / (z_norms + 1e-8)

    cos_arc_mat = arc_mat @ arc_mat.T
    cos_z_mat = z_mat @ z_mat.T

    iu, ju = np.triu_indices(n, k=1)
    cos_arc = cos_arc_mat[iu, ju]
    cos_z = cos_z_mat[iu, ju]

    types_a = np.array([p["type"] for p in pairs])
    targets_a = np.array([p["target_id"] for p in pairs])

    type_combo = np.where(
        types_a[iu] == types_a[ju],
        np.where(types_a[iu] == "swap", "swap_x_swap", "reenact_x_reenact"),
        "swap_x_reenact",
    )
    same_id = (targets_a[iu] == targets_a[ju])

    print(f"pair-pair combos: {len(iu):,}")

    md = []
    md.append("# Orthogonality independent verification — across-pair z_perp\n")
    md.append("Setup: for each FF++ pair P_i (target identity + frame + manipulation), "
              "we have ONE z_perp value (real and fake are identical by OPA construction). "
              "We compare every pair-pair combo (i, j) on:\n")
    md.append("  - **cos_arc** = cos(arc_real_Pi, arc_real_Pj) — independent ArcFace identity similarity")
    md.append("  - **cos_z**   = cos(z_perp_Pi, z_perp_Pj) — DINOv3+OPGTL manipulation-orthogonal subspace similarity\n")
    md.append(f"Pairs loaded: **{n}**  ⟶  pair-pair combos: **{len(iu):,}**\n")

    overall_p = pearson(cos_z, cos_arc)
    overall_s = spearman(cos_z, cos_arc)
    md.append(f"## Overall correlation (all pair-pair combos)\n")
    md.append(f"- Pearson r(cos_z, cos_arc) = **{overall_p:+.4f}**")
    md.append(f"- Spearman ρ(cos_z, cos_arc) = **{overall_s:+.4f}**\n")

    md.append(f"## Per type-combo correlation\n")
    md.append(f"| type combo | n | mean cos_arc | mean cos_z | Pearson r |")
    md.append(f"|---|---|---:|---:|---:|")
    for combo in ("swap_x_swap", "swap_x_reenact", "reenact_x_reenact"):
        mask = type_combo == combo
        if mask.sum() < 2:
            continue
        a, z = cos_arc[mask], cos_z[mask]
        md.append(f"| {combo.replace('_', ' ')} | {mask.sum():,} | "
                  f"{a.mean():+.4f} | {z.mean():+.4f} | {pearson(z, a):+.4f} |")
    md.append("")

    md.append(f"## Same-identity vs different-identity (the headline test)\n")
    md.append("Since FF++ pairs are keyed by target identity (`video_id`), every pair-pair combo "
              "is either *same-identity* (same target across two manipulation methods on the same "
              "person/frame) or *different-identity*. If z_perp preserves the identity manifold, "
              "**same-id should have substantially higher cos_z than different-id**.\n")
    md.append(f"| partition | n | mean cos_arc | mean cos_z |")
    md.append(f"|---|---|---:|---:|")
    for label, mask in [("same target identity", same_id), ("different target identity", ~same_id)]:
        if mask.sum() < 1:
            continue
        md.append(f"| {label} | {mask.sum():,} | "
                  f"{cos_arc[mask].mean():+.4f} | {cos_z[mask].mean():+.4f} |")
    md.append("")

    sep_arc = cos_arc[same_id].mean() - cos_arc[~same_id].mean()
    sep_z = cos_z[same_id].mean() - cos_z[~same_id].mean()
    md.append(f"### Identity separation (mean same-id − mean different-id)")
    md.append(f"- ArcFace: Δ = **{sep_arc:+.4f}** (large positive = identity correctly separated by ArcFace, as expected)")
    md.append(f"- z_perp:  Δ = **{sep_z:+.4f}** (large positive = z_perp also separates identity → orthogonality claim holds)\n")

    md.append(f"## Interpretation\n")
    md.append("- A large positive Δ for z_perp + a positive overall Pearson r implies z_perp encodes "
              "identity in a way that an independently-trained face-recognition model agrees with — "
              "answering R2's circularity charge directly.")
    md.append("- A z_perp Δ near zero implies z_perp does **not** preserve identity structure, "
              "even though it's defined as the manipulation-orthogonal subspace. That would weaken "
              "the §4.6 narrative regardless of the within-pair degeneracy.")
    md.append("- The per-type-combo breakdown lets reviewers see whether the correlation is "
              "driven by swap pairs, reenactment pairs, or both.\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {args.out_md}")

    if args.out_csv_sample:
        rng = np.random.default_rng(args.seed)
        k = min(args.sample_rows, len(iu))
        idx = rng.choice(len(iu), size=k, replace=False)
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv_sample)) or ".", exist_ok=True)
        with open(args.out_csv_sample, "w") as f:
            w = csv.writer(f)
            w.writerow(["i", "j", "target_i", "target_j", "type_combo",
                        "same_identity", "cos_arc", "cos_z"])
            for s in idx:
                ii, jj = int(iu[s]), int(ju[s])
                w.writerow([ii, jj, pairs[ii]["target_id"], pairs[jj]["target_id"],
                            str(type_combo[s]), bool(same_id[s]),
                            f"{cos_arc[s]:.6f}", f"{cos_z[s]:.6f}"])
        print(f"wrote {args.out_csv_sample} ({k} rows sampled from {len(iu):,})")

    print()
    print(f"Overall Pearson r = {overall_p:+.4f}, Spearman rho = {overall_s:+.4f}")
    print(f"Identity separation: ArcFace Δ={sep_arc:+.4f}, z_perp Δ={sep_z:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
