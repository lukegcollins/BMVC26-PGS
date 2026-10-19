# SLERP rerun — MEAN-17 video AUC (%)

| group | SLERP on | SLERP off | matched Δ on−off (pp) | April (SLERP inactive) |
|---|---|---|---|---|
| ln | 91.64 ± 0.12 (n=3) | 91.51 ± 0.07 (n=3) | 0.13 ± 0.07 (n=3) | 91.97 ± 0.10 (n=3) |
| lora | 92.18 ± 0.10 (n=3) | 92.64 ± 0.42 (n=3) | -0.46 ± 0.32 (n=3) | 92.62 ± 0.33 (n=3) |
| gsd_lora | 92.40 ± 0.45 (n=3) | 92.89 ± 0.38 (n=3) | -0.49 ± 0.30 (n=3) | 92.82 ± 0.35 (n=3) |
| ln_opa_unif | — | — | — | 90.90 ± 0.12 (n=3) |
| ln_no_hardrr | — | — | — | 91.76 ± 0.11 (n=3) |
| ln_no_small | — | — | — | 91.69 ± 0.04 (n=3) |
| lora_no_gtl | — | — | — | 92.21 ± 0.44 (n=3) |
| gsd_lora_vbar | — | — | — | 91.80 (n=1) |

Ablation groups (ln_opa_unif, ln_no_hardrr, ln_no_small, lora_no_gtl, gsd_lora_vbar) were dropped from stage 2 under prereg amendment A2 (2026-10-12) before any stage-2 number existed; their rows carry only the April reference (SLERP inactive).
