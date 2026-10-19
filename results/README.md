# results

Evidence behind the paper's tables and figures. Files are copied byte-for-byte from the release and analysis archives, except where a note below says otherwise.

- `logs/`: training and test logs for the nine headline runs and the component ablations. Each run holds its resolved config, a hardware fingerprint, TensorBoard scalars and per-checkpoint test results. The release copy of the logs omitted the `train.log` and `test.log` files, so those were taken from the supplementary archive at the same paths.
- `logs/main/`: the nine headline runs (LN, LoRA and GSD-LoRA at seeds 42, 137 and 1729).
- `logs/ablations/`: the component ablations, the frozen-probe control and the hyperparameter sweep stubs.
- `arcface/`: identity-space verification artefacts for supplementary Appendix G (pair manifest, per-pair similarity table, pair-pair sample and orthogonality summary). The two `.npz` embedding matrices are release assets and are not kept in git.
- `baselines/`: the unified re-evaluation of the external baselines, with `comparison_master.csv` as the method-level summary.
- `baselines/collated/`: `comparison_detail.csv`, the per-dataset detail behind the summary.
- `baselines/raw/`: one summary table per external baseline, from its re-evaluation.
- `repr_geometry/`: the representation-geometry summary (RQ2), written by `src/repr_geometry/repr_geometry.py` from the DeepSpeak-v2 features of the frozen and adapted runs.
- `training_dynamics/`: validation curves and an overfitting summary, taken from the headline runs' training logs.
