#!/usr/bin/env python3
"""
repr_geometry.py — RQ2 representation-geometry battery, regenerated from
ALREADY-EXTRACTED DSv2 features (no GPU / no model load).

Definitions follow the original tooling:
  - V_K identity subspace = PCA on L2-normalised, per-video mean-pooled
    frozen-DINOv3 CLS embeddings              (tools/otg/fit_identity_pca.py)
  - identity decontamination z_perp = z - V_K V_K^T z   (tools/otg/otg_extractor.py)
  - OPA pairwise residual orthogonal to v=unit(Delta)   (extract_zperp.py)

Reports several coupling formulations and a held-out nearest-prototype test.

Features on disk:
  RAW (frozen)   : frozen_probe_s42 / epoch24_best_macro / DSv2
  ADAPTED        : gsd_lora_s42    / epoch07_best_macro / DSv2
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ADAPTED = ROOT / "results/logs/main/orthoflow_gtl_gsd_lora_s42/test/epoch07_best_macro/per_dataset/DSv2/features.npz"
RAW = ROOT / "results/logs/ablations/orthoflow_gtl_frozen_probe_s42/test/epoch24_best_macro/per_dataset/DSv2/features.npz"
MANIFEST = ROOT / "data/manifests/DeepSpeak-v2/256w_256h_1.3x_32f_insightface/80-20-T/items.jsonl"
OUTDIR = ROOT / "results/repr_geometry"
K_SWEEP = [64, 128, 256, 512, 768]
SIX = ["diff2lip", "latentsync", "memo", "hellomeme", "facefusion", "liveportrait"]
RNG = np.random.default_rng(42)


def per_video(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    vids = d["video_ids"].astype(str)
    feats = d["features"].astype(np.float64)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)  # per-frame L2 (matches fit_identity_pca)
    labels = d["labels"].astype(int)
    sums, counts, lab = {}, {}, {}
    for v, f, l in zip(vids, feats, labels):
        if v in sums:
            sums[v] += f; counts[v] += 1
        else:
            sums[v] = f.copy(); counts[v] = 1; lab[v] = int(l)
    vec = {v: sums[v] / counts[v] for v in sums}      # per-video mean of unit frames
    return vec, lab


def load_pairing():
    target_of = {}
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            if (it.get("label") or {}).get("class_id") != 1:
                continue
            a = it.get("attributes", {}) or {}
            t, r = a.get("identity_target"), a.get("recording_target")
            if t is not None and r is not None:
                target_of[it["key"]] = f"{t}-{r}"
    return target_of


def build_pairs(vec, lab, target_of):
    real_by_key = {v.split("/")[-1]: v for v, l in lab.items() if l == 0}
    pairs = []
    for v, l in lab.items():
        if l != 1:
            continue
        tgt = target_of.get(v.split("/")[-1])
        if tgt is None or tgt not in real_by_key:
            continue
        pairs.append((vec[real_by_key[tgt]], vec[v], v.split("/")[-2]))
    return pairs


def fit_VK(real_mat):
    mu = real_mat.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(real_mat - mu, full_matrices=False)
    return Vt, mu[0]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def opa_zperp(z_r, z_f, eps=1e-8):
    """extract_zperp.py opa_residual: residuals orthogonal to v=unit(Delta).
    Returns the UN-normalised residuals; by OPA construction zpr_raw == zpf_raw
    (the manipulation direction is removed), so ||zpf_raw - zpr_raw|| ~ machine eps."""
    delta = z_f - z_r
    v = delta / (np.linalg.norm(delta) + eps)
    zpr = z_r - (z_r @ v) * v
    zpf = z_f - (z_f @ v) * v
    return zpr, zpf, v


def metrics(pairs, Vt, mu, K):
    Vk = Vt[:K]; pc1 = Vt[0]
    m = {k: [] for k in ["cos_zf_zr", "id_energy", "cosD_zrId", "cosD_mu", "cosD_pc1",
                          "opa_resid_diff"]}
    for z_r, z_f, _ in pairs:
        d = z_f - z_r
        nd = np.linalg.norm(d)
        if nd < 1e-12:
            continue
        d_id = Vk.T @ (Vk @ d)
        z_r_id = Vk.T @ (Vk @ z_r)
        m["cos_zf_zr"].append(cos(z_f, z_r))
        m["id_energy"].append(float((np.linalg.norm(d_id) / nd) ** 2))     # frac of ||D||^2 in identity subspace
        m["cosD_zrId"].append(cos(d, z_r_id))                              # signed: displacement vs real's identity comp
        m["cosD_mu"].append(cos(d, mu))                                    # signed: displacement vs mean-real direction
        m["cosD_pc1"].append(cos(d, pc1))                                  # signed: displacement vs PC1
        zpr, zpf, v = opa_zperp(z_r, z_f)
        m["opa_resid_diff"].append(float(np.linalg.norm(zpf - zpr)))      # OPA by-construction ~0 (machine eps)
    return {k: np.array(val) for k, val in m.items()}


def prototype_loo(pairs, Vt, K, folds=5):
    """Held-out nearest-prototype attack-type accuracy: raw Delta dir vs identity-orthogonal residual."""
    Vk = Vt[:K]
    raw, perp, meth = [], [], []
    for z_r, z_f, mname in pairs:
        if mname not in SIX:
            continue
        d = z_f - z_r; nd = np.linalg.norm(d)
        if nd < 1e-12:
            continue
        dp = d - Vk.T @ (Vk @ d); npd = np.linalg.norm(dp)
        if npd < 1e-12:
            continue
        raw.append(d / nd); perp.append(dp / npd); meth.append(mname)
    raw = np.stack(raw); perp = np.stack(perp); meth = np.array(meth)
    n = len(meth); idx = RNG.permutation(n); fold_id = idx % folds

    def cv(X):
        correct = 0
        for f in range(folds):
            te = fold_id == f; tr = ~te
            protos, keys = [], []
            for mm in SIX:
                sel = tr & (meth == mm)
                if sel.sum() == 0:
                    continue
                p = X[sel].mean(0); p /= (np.linalg.norm(p) + 1e-12)
                protos.append(p); keys.append(mm)
            P = np.stack(protos)
            pred = [keys[i] for i in (X[te] @ P.T).argmax(1)]
            correct += sum(p == m for p, m in zip(pred, meth[te]))
        return correct / n
    return {"n": n, "raw_acc": cv(raw), "perp_acc": cv(perp), "chance": 1.0 / len(SIX)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    target_of = load_pairing()
    out = ["# RQ2 representation-geometry battery (regenerated, no GPU)", ""]
    for tag, path in [("RAW (frozen_probe s42)", RAW), ("ADAPTED (gsd_lora s42 ep07)", ADAPTED)]:
        vec, lab = per_video(path)
        pairs = build_pairs(vec, lab, target_of)
        real_mat = np.stack([vec[v] for v, l in lab.items() if l == 0])
        Vt, mu = fit_VK(real_mat)
        out += [f"## {tag}", f"pairs={len(pairs)}  reals={len(real_mat)}", "",
                "| K | cos(z_f,z_r) | id-energy(ΔinV_K) | cos(Δ,z_r^id) | cos(Δ,μ) | cos(Δ,PC1) | proto raw→perp |",
                "|---|---|---|---|---|---|---|"]
        for K in K_SWEEP:
            if K > Vt.shape[0]:
                continue
            mm = metrics(pairs, Vt, mu, K)
            pa = prototype_loo(pairs, Vt, K)
            out.append(f"| {K} | {mm['cos_zf_zr'].mean():.3f} | {mm['id_energy'].mean():.3f} | "
                       f"{mm['cosD_zrId'].mean():+.3f} | {mm['cosD_mu'].mean():+.3f} | {mm['cosD_pc1'].mean():+.3f} | "
                       f"{pa['raw_acc']*100:.1f}→{pa['perp_acc']*100:.1f}% |")
        mm = metrics(pairs, Vt, mu, min(512, Vt.shape[0]))
        rd = mm['opa_resid_diff']
        out.append("")
        out.append(f"OPA residual ||z_perp^f - z_perp^r||: mean={rd.mean():.3e}, p95={np.percentile(rd,95):.3e} "
                   f"(should be ~machine-eps: real/fake residuals identical by construction)")
        out.append(f"chance proto={100/len(SIX):.1f}%")
        out.append("")
    txt = "\n".join(out)
    (OUTDIR / "repr_geometry_summary.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
