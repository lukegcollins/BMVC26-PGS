# RQ2 representation-geometry battery (regenerated, no GPU)

## RAW (frozen_probe s42)
pairs=1416  reals=1863

| K | cos(z_f,z_r) | id-energy(ΔinV_K) | cos(Δ,z_r^id) | cos(Δ,μ) | cos(Δ,PC1) | proto raw→perp |
|---|---|---|---|---|---|---|
| 64 | 0.939 | 0.454 | -0.147 | +0.008 | +0.037 | 66.9→73.2% |
| 128 | 0.939 | 0.658 | -0.147 | +0.008 | +0.037 | 67.4→79.4% |
| 256 | 0.939 | 0.792 | -0.152 | +0.008 | +0.037 | 67.7→84.0% |
| 512 | 0.939 | 0.907 | -0.154 | +0.008 | +0.037 | 67.5→83.8% |
| 768 | 0.939 | 0.966 | -0.154 | +0.008 | +0.037 | 67.2→75.4% |

OPA residual ||z_perp^f - z_perp^r||: mean=2.000e-08, p95=2.000e-08 (should be ~machine-eps: real/fake residuals identical by construction)
chance proto=16.7%

## ADAPTED (gsd_lora s42 ep07)
pairs=1416  reals=1863

| K | cos(z_f,z_r) | id-energy(ΔinV_K) | cos(Δ,z_r^id) | cos(Δ,μ) | cos(Δ,PC1) | proto raw→perp |
|---|---|---|---|---|---|---|
| 64 | 0.380 | 0.730 | -0.591 | -0.327 | -0.286 | 39.5→65.7% |
| 128 | 0.380 | 0.810 | -0.605 | -0.327 | -0.286 | 38.6→69.4% |
| 256 | 0.380 | 0.878 | -0.613 | -0.327 | -0.286 | 38.7→70.8% |
| 512 | 0.380 | 0.944 | -0.617 | -0.327 | -0.286 | 38.3→71.9% |
| 768 | 0.380 | 0.980 | -0.617 | -0.327 | -0.286 | 39.2→66.6% |

OPA residual ||z_perp^f - z_perp^r||: mean=2.000e-08, p95=2.000e-08 (should be ~machine-eps: real/fake residuals identical by construction)
chance proto=16.7%
