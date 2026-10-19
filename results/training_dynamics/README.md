# Training dynamics

- `val_curves.csv`: per-epoch validation loss and Video-AUC for every run and criterion (dsv2, cdfv3, ffsh, macro), parsed from the training logs.
- `overfitting_summary.csv`: per-run peak epoch, peak and final macro AUC, overfit gap, plateau width and post-peak slope, written by `src/dynamics/parse_train_logs.py`.

## Post-peak slope

Table 5 fits the post-peak slope from the validation peak through the final epoch (24), as the paper describes, using at least 2 points. `src/dynamics/table5.py` does this.

For LN seed 137 the peak is at epoch 23, so the fit uses 2 points and gives -0.0100. This matches the original analysis file `training_stability.csv`, and the three-seed LN mean of -0.0089 prints as -0.009.

The vendored `parse_train_logs.py` requires at least 3 post-peak points. `overfitting_summary.csv` (shipped verbatim) therefore has no slope for LN seed 137, and a mean taken over that file would print -0.008.
