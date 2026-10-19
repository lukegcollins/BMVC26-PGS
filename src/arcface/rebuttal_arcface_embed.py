#!/usr/bin/env python3
"""
rebuttal_arcface_embed.py — extract InsightFace (ArcFace) identity embeddings
for a list of face crops, independent of the DINOv3 space our OPA loss shapes.

Why this exists (R2 rebuttal): the orthogonality verification in Sec. 4.6 of
the paper estimates the identity subspace by PCA on the SAME DINOv3 embedding
space the OPA loss is defined on, so the "near-zero coupling" between
z_perp and the identity axis is partly true by construction. The clean fix
is to use a frozen, off-the-shelf face-recognition model trained on a
different objective (identity classification across millions of faces) as
an independent identity space, then correlate cosine similarity in that
space with cosine similarity in our DINOv3 z_perp residual space.

This script does the first half: extract ArcFace embeddings for a flat
manifest of frames. The pairing + correlation analysis lives in the
sibling `rebuttal_arcface_correlate.py` script.

Dependencies
    pip install insightface onnxruntime-gpu numpy opencv-python tqdm

Usage
    python rebuttal_arcface_embed.py \
        --manifest /path/to/frames.jsonl \
        --out /path/to/arcface_embeddings.npz \
        [--device cuda] [--ctx-id 0] [--batch-size 64]

Manifest format (JSONL, one entry per line)
    {"frame_id": "ff_df_001_frame_00012",
     "frame_path": "/path/to/frame_00012.jpg",
     "video_id": "001", "manipulation": "FF-DF", "label": "fake"}

The frame_id is opaque to this script; only frame_path is read. The other
fields are passed through unchanged to the output so downstream pairing
can match real-fake pairs by (video_id, frame_index, manipulation).

Output
    A single NPZ archive at --out containing:
        ids:         (N,)      object array of frame_id strings
        embeddings:  (N, 512)  float32 L2-normalised ArcFace embeddings
        metadata:    (N,)      object array of JSON-serialised metadata dicts
                                from the manifest (everything except frame_path)
        valid:       (N,)      bool — False if no face was detected in the frame
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    sys.exit("missing opencv-python — pip install opencv-python")

try:
    from insightface.app import FaceAnalysis
except ImportError as exc:
    sys.exit("missing insightface — pip install insightface onnxruntime-gpu")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_):  # type: ignore
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="JSONL manifest of frames")
    p.add_argument("--out", required=True, help="Output NPZ path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--ctx-id", type=int, default=0,
        help="ONNX runtime context id (GPU index for cuda; -1 forces CPU)",
    )
    p.add_argument(
        "--det-size", type=int, default=640,
        help="Face detector input size (square). 640 matches buffalo_l default.",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N entries (0 = all). Useful for smoke tests.",
    )
    return p.parse_args()


def load_face_analysis(ctx_id: int, det_size: int) -> FaceAnalysis:
    """Load buffalo_l with detection + recognition models active."""
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def embed_frame(app: FaceAnalysis, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Run detection + recognition; return the largest face's L2-normalised embedding."""
    faces = app.get(frame_bgr)
    if not faces:
        return None
    # Take the largest face (the deepfake crop should occupy most of the frame)
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    emb = faces[0].normed_embedding.astype(np.float32)
    return emb


def main() -> int:
    args = parse_args()
    if args.device == "cpu":
        args.ctx_id = -1

    print(f"Loading buffalo_l (det+rec) on ctx_id={args.ctx_id}, det_size={args.det_size}")
    app = load_face_analysis(args.ctx_id, args.det_size)

    ids: list[str] = []
    metas: list[dict] = []
    embeds: list[np.ndarray] = []
    valid: list[bool] = []

    n_total = 0
    n_valid = 0
    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        entries = entries[: args.limit]

    for e in tqdm(entries, desc="embedding"):
        frame_path = e["frame_path"]
        frame_id = e["frame_id"]
        meta = {k: v for k, v in e.items() if k not in ("frame_path",)}
        n_total += 1
        if not os.path.exists(frame_path):
            ids.append(frame_id); metas.append(meta); valid.append(False); embeds.append(np.zeros(512, np.float32))
            continue
        img = cv2.imread(frame_path)
        if img is None:
            ids.append(frame_id); metas.append(meta); valid.append(False); embeds.append(np.zeros(512, np.float32))
            continue
        emb = embed_frame(app, img)
        if emb is None:
            ids.append(frame_id); metas.append(meta); valid.append(False); embeds.append(np.zeros(512, np.float32))
            continue
        ids.append(frame_id); metas.append(meta); valid.append(True); embeds.append(emb)
        n_valid += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(
        args.out,
        ids=np.asarray(ids, dtype=object),
        embeddings=np.stack(embeds, axis=0),
        metadata=np.asarray([json.dumps(m) for m in metas], dtype=object),
        valid=np.asarray(valid, dtype=bool),
    )
    print(f"wrote {args.out}: {n_valid}/{n_total} frames embedded successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
