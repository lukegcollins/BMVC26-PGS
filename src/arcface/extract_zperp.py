#!/usr/bin/env python3
"""
extract_zperp.py — produce zperp_embeddings.npz from a trained OrthoflowGTL
checkpoint, paired with the same FF++ manifest used by rebuttal_arcface_embed.py.

This script is the second-half companion to rebuttal_arcface_embed.py: where
that script extracts independent ArcFace identity embeddings, this one extracts
the OPA orthogonal residuals z_perp from our trained model on the SAME frames.
The downstream rebuttal_arcface_correlate.py then pairs the two and reports
the correlation that addresses R2's circularity concern in Section 4.6 of the
paper.

Why this lives in 04_results/scripts/ and not 03_source_code/:
    The OPA pairwise math is small and well-defined; we keep it here so the
    rebuttal verification stays a self-contained artefact rather than touching
    the training-side detector code.

Usage
    python src/arcface/extract_zperp.py \
        --checkpoint 04_results/logs/bmvc_experiments/orthoflow_gtl_gsd_lora_s42/checkpoints/epoch=07-video_auc=0.8678.ckpt \
        --config     03_source_code/configs/experiments/orthopair-gtl/lora_gtl_gsd/orthoflow_gtl_gsd_lora_s42.yaml \
        --manifest   results/arcface/ff_pair_manifest.jsonl \
        --out        data/arcface/zperp_embeddings.npz \
        [--device cuda] [--batch-size 32]

Output NPZ schema (matches what rebuttal_arcface_correlate.py expects)
    ids:        (N,)    object       — frame_id strings (same as manifest)
    z_perp:     (N, d)  float32      — unit-norm OPA residuals (d = 1024 for DINOv3 ViT-L/16)
    metadata:   (N,)    object       — JSON-serialised manifest entries, minus frame_path
    valid:      (N,)    bool         — True where z_perp is meaningful (face detected, pair complete)

Math (canonical; mirrors src/loss/_orthoflow_gtl_core.py::OrthoPairGTLCore._orthogonal_alignment)
    For each paired (z_real, z_fake) in R^d, both already L2-normalised by the
    DINOv3 backbone (backbone_config.l2_normalize: true in all our configs):

        delta = z_fake - z_real
        v = delta / (||delta|| + eps)            # unit displacement
        alpha_r = z_real · v
        alpha_f = z_fake · v
        z_perp_real_raw = z_real - alpha_r * v
        z_perp_fake_raw = z_fake - alpha_f * v
        z_perp_real = z_perp_real_raw / (||z_perp_real_raw|| + eps)
        z_perp_fake = z_perp_fake_raw / (||z_perp_fake_raw|| + eps)

    z_perp is therefore the unit-norm component of each embedding that is
    orthogonal to the pair's manipulation direction v.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
    import cv2
except ImportError as e:
    sys.exit(f"missing dependency: {e}. install torch + opencv-python.")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_): return x


# Make the training source importable when it sits next to the repository root.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SRC_DIR = PROJECT_ROOT / "03_source_code"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


def load_model(ckpt_path: str, config_path: str, device: torch.device):
    """Load a trained OrthoflowGTL model from a Lightning .ckpt + matching config.

    The project may expose a higher-level loader (e.g.
    src.detectors.registry.build_from_config, OrthoflowDetector.load_from_checkpoint,
    or a Lightning Trainer test-time path). If so, prefer that loader and have
    it return an object with a .backbone callable that maps (B, 3, H, W) tensors
    to (B, d) L2-normalised DINOv3 features.

    The fallback below assumes a straightforward Lightning load and that the
    detector module exposes `model.backbone(x) -> (B, d)`.
    """
    cfg = yaml.safe_load(open(config_path))

    # Try the project's high-level loader first; fall back to plain Lightning.
    try:
        from src.detectors.orthoflow_detector import OrthoflowDetector
        model = OrthoflowDetector.load_from_checkpoint(
            ckpt_path, map_location=device, strict=False,
        )
    except Exception:
        # Manual reconstruction path — adapt as needed for the project's API.
        import pytorch_lightning as pl
        model = pl.LightningModule.load_from_checkpoint(
            ckpt_path, map_location=device, strict=False,
        )
    model.eval()
    model.to(device)
    return model, cfg


def preprocess(frame_path: str, mean, std, size=224):
    """Load a 256x256 face-crop PNG, normalise, return CHW tensor."""
    img = cv2.imread(frame_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (img.shape[0], img.shape[1]) != (size, size):
        img = cv2.resize(img, (size, size))
    arr = img.astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))


def opa_residual(z_real: torch.Tensor, z_fake: torch.Tensor, eps: float = 1e-8):
    """Compute OPA orthogonal residuals for a paired batch. Inputs (B, d)."""
    delta = z_fake - z_real
    v = delta / (delta.norm(dim=-1, keepdim=True) + eps)
    alpha_r = (z_real * v).sum(dim=-1, keepdim=True)
    alpha_f = (z_fake * v).sum(dim=-1, keepdim=True)
    z_perp_real_raw = z_real - alpha_r * v
    z_perp_fake_raw = z_fake - alpha_f * v
    z_perp_real = z_perp_real_raw / (z_perp_real_raw.norm(dim=-1, keepdim=True) + eps)
    z_perp_fake = z_perp_fake_raw / (z_perp_fake_raw.norm(dim=-1, keepdim=True) + eps)
    return z_perp_real, z_perp_fake


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt")
    p.add_argument("--config", required=True, help="Matching config YAML for the run")
    p.add_argument("--manifest", required=True, help="JSONL manifest (same one passed to rebuttal_arcface_embed.py)")
    p.add_argument("--out", required=True, help="Output NPZ path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"device: {device}")

    print(f"loading model from {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device)
    mean = cfg.get("backbone_config", {}).get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("backbone_config", {}).get("std", [0.229, 0.224, 0.225])
    size = int(cfg.get("resolution", 224))
    print(f"preprocessing: size={size}, mean={mean}, std={std}")

    # Load manifest
    entries = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"manifest: {len(entries)} entries")

    # Group by pair key (video_id, frame_index, manipulation)
    pair_index = {}
    for e in entries:
        key = (e["video_id"], e["frame_index"], e["manipulation"])
        pair_index.setdefault(key, {})[e["label"]] = e
    pairs = [(k, v) for k, v in pair_index.items() if "real" in v and "fake" in v]
    print(f"complete real-fake pairs: {len(pairs)}")

    # Embed every unique frame once
    unique_entries = list({e["frame_id"]: e for e in entries}.values())
    z_by_id = {}
    with torch.no_grad():
        for start in tqdm(range(0, len(unique_entries), args.batch_size), desc="embedding"):
            chunk = unique_entries[start:start + args.batch_size]
            tensors, ids = [], []
            for e in chunk:
                t = preprocess(e["frame_path"], mean, std, size=size)
                if t is None:
                    continue
                tensors.append(t)
                ids.append(e["frame_id"])
            if not tensors:
                continue
            batch = torch.stack(tensors).to(device)
            # Adjust to whatever attribute exposes the (B,d) L2-normalised
            # DINOv3 features. Common patterns:
            #   feats = model.backbone(batch)
            #   feats = model.encoder(batch)
            #   feats, _ = model.forward_features(batch)
            feats = model.backbone(batch)
            if hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state.mean(dim=1)  # if backbone returns HF-style output
            feats = feats.detach().cpu().numpy()
            for fid, z in zip(ids, feats):
                z_by_id[fid] = z

    print(f"embedded {len(z_by_id)} / {len(unique_entries)} unique frames")

    # Pairwise OPA decomposition
    ids_out, z_perp_out, meta_out, valid_out = [], [], [], []
    skipped = 0
    for (vid, fidx, manip), pair in tqdm(pairs, desc="opa"):
        r_id = pair["real"]["frame_id"]
        f_id = pair["fake"]["frame_id"]
        if r_id not in z_by_id or f_id not in z_by_id:
            skipped += 1
            continue
        z_r = torch.tensor(z_by_id[r_id]).unsqueeze(0)
        z_f = torch.tensor(z_by_id[f_id]).unsqueeze(0)
        zpr, zpf = opa_residual(z_r, z_f)
        for fid, zp, e in [
            (r_id, zpr[0], pair["real"]),
            (f_id, zpf[0], pair["fake"]),
        ]:
            ids_out.append(fid)
            z_perp_out.append(zp.numpy().astype(np.float32))
            m = {k: v for k, v in e.items() if k != "frame_path"}
            meta_out.append(json.dumps(m))
            valid_out.append(True)

    if skipped:
        print(f"warning: skipped {skipped} pairs (one or both embeddings missing)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(
        args.out,
        ids=np.asarray(ids_out, dtype=object),
        z_perp=np.stack(z_perp_out).astype(np.float32),
        metadata=np.asarray(meta_out, dtype=object),
        valid=np.asarray(valid_out, dtype=bool),
    )
    print(f"wrote {args.out}: {len(ids_out)} z_perp embeddings from {len(pairs) - skipped} pairs")

    # Quick sanity: unit norm
    arr = np.stack(z_perp_out)
    norms = np.linalg.norm(arr, axis=1)
    print(f"z_perp norm: min={norms.min():.6f}, max={norms.max():.6f} (should be ≈ 1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
