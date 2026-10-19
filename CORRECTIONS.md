# BMVC26-PGS: corrections after camera-ready

## 2026-10-19: paired SLERP removed from the method description; ablation labels

**Before.** Git tag `bmvc26-camera-ready` (commit `83fb53a`), `paper_camera_ready.pdf` and
`paper_camera_ready_supp.pdf`, and the supplementary archive uploaded with the paper (its `*.log`
files are carried in `results/logs/`). The camera-ready PDFs are not in this repository.

**After.** The corrected build, `pdfs/paper.pdf` and `pdfs/supplementary.pdf` (17 + 22 pp; the body
now ends on p13, and the References start on p14), together with the release tree in this
repository: `configs/`, `results/` and `pdfs/`.

**Why.** The paired SLERP augmentation described in §3 (Manifold Regularisation) was not active in
any reported run:
- The interpolation code had been removed from the training code base on 2026-03-11.
- Its flag (`use_paired_slerp_aug`) and log line ("Use SLERP aug: True") remained, so every
  headline and ablation run trained with uniformity on the observed embeddings and OPA residuals only.

**What this means for the numbers.**
- **No number changes.** The reported numbers are correct for the model that was trained, and
  this correction makes the text describe that model.
- **Matched three-seed rerun with the interpolation restored as described** (the verdict is
  `publication/SLERP_Rerun/results/slerp_rerun_verdict.md` in the research workbench repository).
  On minus off, MEAN-17 changes by +0.13 (LN), −0.46 (LoRA) and −0.49 (GSD-LoRA) pp, all under the
  registered 0.5 pp threshold. With it off, the rerun gives 91.51 / 92.64 / 92.89 (reported:
  91.97 / 92.62 / 92.82).

### Main paper (paper source, `sections/`)

| where | before | after |
|---|---|---|
| `02_introduction.tex` (contribution bullet, Manifold Regularisation) | "we use paired Spherical Linear Interpolation (SLERP) augmentation together with a hyperspherical uniformity objective." | "we apply a hyperspherical uniformity objective to the batch embeddings and to the orthogonal residuals of each pair." |
| `03_related_work.tex` (after LSDA) | "we also augment the feature space, but we generate interpolations anchored to aligned real–fake endpoints, so synthesised samples follow coherent Real→Fake transformations rather than unconstrained perturbations." | "rather than synthesising feature-space samples, we constrain the geometry of aligned real–fake embeddings, so the representation encodes coherent Real→Fake transformations rather than invariance to unconstrained perturbations." |
| `04_methodology.tex` §3 intro | "…uniformity objective with two extensions: paired SLERP augmentation densifies the manifold between aligned endpoints, and residual uniformity spreads…" | "…uniformity objective to the batch embeddings, extended by residual uniformity, which spreads…" |
| `04_methodology.tex` after Eq. (unif) | "N is the number of batch embeddings (paired samples plus their SLERP interpolations)" | "N is the number of batch embeddings" |
| `04_methodology.tex` | paragraph "Paired SLERP Augmentation" with Eq. (slerp) and the LSDA contrast sentence | removed (no `\ref{eq:slerp}` existed) |
| `05_experiments.tex` RQ3 | "(OPA, hyperspherical uniformity + SLERP, GTL, GSD)" | "(OPA, hyperspherical uniformity, GTL, GSD)" |

### Ablation labels (Table 6, `table3_unified.tex`; supp. Table S11, `supplementary/table3_sensitivity.tex`)

- **The issue.** The "Full method" rows also train with uniformity (β = 0.5 plus residual
  uniformity; see the configs), but the labels omitted it.
  - Values are unchanged.
  - The ladder reads OPA → OPA + uniformity → OPA + uniformity + GTL.
- **Label changes:**
  - "+ OPA + GTL (Full method)" → "+ OPA + uniformity + GTL (Full method)". This applies to both
    tables; S11 also carries ", anchor".
  - "+ OPA + GTL" → "+ OPA + uniformity + GTL" (LoRA tier).
  - "+ OPA + GTL + GSD (GSD-LoRA)" → "+ OPA + uniformity + GTL + GSD (GSD-LoRA)".
- **Matching prose in `05_experiments.tex`** (component-ablation paragraph):
  - "LN+OPA+GTL" → "LN+OPA+uniformity+GTL" (twice).
  - "uniformity is harmful and GTL on top of OPA is marginally net-negative" → "uniformity is
    harmful, and uniformity plus GTL on top of OPA is marginally net-negative (92.36→91.97)".
  - LoRA tier: "uniformity is roughly neutral (92.24→92.21)" → "OPA with uniformity is roughly
    neutral (92.24→92.21)". There is no LoRA + OPA-only row, so the step is OPA + uniformity
    against no geometry.

### Posters (outside this repository)

- gemini: "paired SLERP augmentation and hyperspherical uniformity" → "hyperspherical uniformity on
  the embeddings and residuals".
- betterposter: "SLERP + uniformity" → "Uniformity".

### Release (`configs/`, `results/`, `pdfs/`)

- `configs/`: the configs change `use_paired_slerp_aug: true` → `false`. An inline comment marks the
  change in 34 of the 46 files, and `configs/ablations/_generate.py` is changed to match. `results/logs/` keep the flag as
  recorded.
- The release README: "SLERP + hyperspherical uniformity" → "hyperspherical uniformity"; new section
  "Note on paired SLERP" (what happened, what ran, the rerun numbers).
- `reproduce_tables.py`: same ablation labels as Table 6; label column widened.
