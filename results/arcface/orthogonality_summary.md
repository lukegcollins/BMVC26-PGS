# Orthogonality independent verification — across-pair z_perp

Setup: for each FF++ pair P_i (target identity + frame + manipulation), we have ONE z_perp value (real and fake are identical by OPA construction). We compare every pair-pair combo (i, j) on:

  - **cos_arc** = cos(arc_real_Pi, arc_real_Pj) — independent ArcFace identity similarity
  - **cos_z**   = cos(z_perp_Pi, z_perp_Pj) — DINOv3+OPGTL manipulation-orthogonal subspace similarity

Pairs loaded: **1820**  ⟶  pair-pair combos: **1,655,290**

## Overall correlation (all pair-pair combos)

- Pearson r(cos_z, cos_arc) = **+0.1109**
- Spearman ρ(cos_z, cos_arc) = **+0.0204**

## Per type-combo correlation

| type combo | n | mean cos_arc | mean cos_z | Pearson r |
|---|---|---:|---:|---:|
| swap x swap | 514,605 | +0.0139 | +0.0527 | +0.0976 |
| swap x reenact | 817,075 | +0.0148 | +0.0531 | +0.1222 |
| reenact x reenact | 323,610 | +0.0138 | +0.0583 | +0.1023 |

## Same-identity vs different-identity (the headline test)

Since FF++ pairs are keyed by target identity (`video_id`), every pair-pair combo is either *same-identity* (same target across two manipulation methods on the same person/frame) or *different-identity*. If z_perp preserves the identity manifold, **same-id should have substantially higher cos_z than different-id**.

| partition | n | mean cos_arc | mean cos_z |
|---|---|---:|---:|
| same target identity | 11,445 | +0.8513 | +0.2317 |
| different target identity | 1,643,845 | +0.0085 | +0.0528 |

### Identity separation (mean same-id − mean different-id)
- ArcFace: Δ = **+0.8428** (large positive = identity correctly separated by ArcFace, as expected)
- z_perp:  Δ = **+0.1789** (large positive = z_perp also separates identity → orthogonality claim holds)

## Interpretation

- A large positive Δ for z_perp + a positive overall Pearson r implies z_perp encodes identity in a way that an independently-trained face-recognition model agrees with — answering R2's circularity charge directly.
- A z_perp Δ near zero implies z_perp does **not** preserve identity structure, even though it's defined as the manipulation-orthogonal subspace. That would weaken the §4.6 narrative regardless of the within-pair degeneracy.
- The per-type-combo breakdown lets reviewers see whether the correlation is driven by swap pairs, reenactment pairs, or both.
