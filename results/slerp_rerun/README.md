# SLERP rerun evidence

This folder holds the evidence for the SLERP rerun reported in the paper. The original BMVC runs were trained with SLERP inactive, so the rerun repeats the three arms (LN, LoRA and GSD-LoRA) with SLERP switched on and off, over three seeds (42, 137 and 1729).

## Files

- `PREREG_2026-09-29_slerp_rerun.md`: the preregistration, copied verbatim from the project record. It includes the registered amendments A1 and A2 and their dates.
- `slerp_rerun_verdict.md`: the verdict produced by the readout, copied verbatim. It gives the MEAN-17 video AUC for each arm and SLERP setting, the matched on-minus-off deltas, and the April reference values.
- `runs.csv`: one row per trained run (18 rows). Columns are `arm` (ln, lora or gsd_lora), `slerp` (on or off), `seed` (42, 137 or 1729) and `mean17`, the MEAN-17 video AUC in percent to four decimal places.

## How the numbers were produced

The rerun used the all-frames data source, as registered in amendment A1 of the preregistration. Each `mean17` value is the mean of the per-dataset video AUC over the 17 datasets in the BMVC MEAN column, taken from the checkpoint that was best on the macro validation score, as scored by the test rescore with the 22-set protocol.

`runs.csv` was built from the rerun logs with the readout functions in the verdict pipeline. The logs are not shipped with this repository, so the file cannot be regenerated from it alone. The verdict and the prereg are the record of the decision rule and of the analysis as run.
