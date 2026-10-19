# results

Evidence behind the paper's tables and figures. Most files are copies, taken byte for byte from the release and analysis archives. Three are derived: they were written for this repository from other records, and the list below says how.

Copies:

- `logs/`: training and test logs for the nine headline runs and the component ablations. Each run holds its resolved config, a hardware fingerprint, TensorBoard scalars and per-checkpoint test results. The release copy of the logs omitted the `train.log` and `test.log` files, so those were taken from the supplementary archive at the same paths.
  - `logs/main/`: the nine headline runs (LN, LoRA and GSD-LoRA at seeds 42, 137 and 1729), and `test_master.csv`, one row per tested checkpoint.
  - `logs/ablations/`: the component ablations at three seeds, the frozen-probe control and the GSD-LoRA `vbar` run.
  - `logs/ablations/sweeps/`: the ten hyperparameter sweeps of Table S12, each a full single-seed run (seed 42).
- `arcface/`: the identity-space check of supplementary Appendix G (Tables S8 and S9): the pair manifest, the per-pair similarity table, the pair-pair sample and the orthogonality summary. `arcface/README.md` says which script wrote each file. The two `.npz` embedding matrices are release assets and are not kept in git.
- `baselines/comparison_master.csv`: the method-level summary of the unified re-evaluation of the external baselines.
  - `baselines/collated/`: `comparison_detail.csv`, the per-dataset detail behind the summary.
  - `baselines/raw/`: one summary table per external baseline, from its re-evaluation.
- `repr_geometry/`: the representation-geometry summary (RQ2), written by `src/repr_geometry/repr_geometry.py` from the DeepSpeak-v2 features of the frozen and adapted runs.
- `slerp_rerun/`: the registration (`PREREG_2026-09-29_slerp_rerun.md`) and the verdict (`slerp_rerun_verdict.md`) of the SLERP rerun, copied verbatim. `slerp_rerun/README.md` describes them.
- `training_dynamics/`: `val_curves.csv` and `overfitting_summary.csv`, parsed from the headline runs' training logs by `src/dynamics/parse_train_logs.py`. The script also writes `train_curves.csv`, which is not kept in git. `training_dynamics/README.md` gives the slope rule of Table 5.

Derived:

- `published/published_numbers.csv`: the baseline values printed in four other works, typed for this repository with the table each value comes from (column `source`).
- `slerp_rerun/runs.csv`: one MEAN-17 value per rerun, built from the rerun logs, which are not shipped.
- `baselines/tex/`: the LaTeX rows of Tables 2, 3, S2, S3 and S5, written by `src/tables/emit_paper_tables.py` and `src/tables/emit_supplementary_tables.py` from `comparison_master.csv` and `published_numbers.csv`. Running the emitters rewrites them unchanged.

A note on the configurations recorded in the logs: the nine headline runs and most ablation and sweep runs record `use_paired_slerp_aug: true` (32 of the 44 `config.yaml` files; the no-geometry, OPA-only and frozen-probe runs record `false`). That flag had no effect in the code that produced these runs, because the interpolation it once switched on had been removed. The configurations in `configs/` set it to `false`, which matches what ran (`method/METHOD.md`, `CORRECTIONS.md`).
