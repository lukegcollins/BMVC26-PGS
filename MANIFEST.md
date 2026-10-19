# MANIFEST

This file maps every figure and table of the paper to the script that produces it, the artefact it reads or writes, and whether it can be regenerated from what this repository ships. Paths are relative to the repository root, and every command is run from there. Each label is followed by the number printed in the paper (Table 3, Figure 2) or in the supplementary (Table S9, Figure S4).

Status values:

| Status | Meaning |
|---|---|
| regenerates from shipped inputs | Run the command from the root; every input is in the repository. |
| regenerates with release assets | As above, but the input includes the two ArcFace arrays, which are release assets. Download them into `data/arcface/` first (`CHECKPOINTS.md` gives the URLs). `src/test_manifest.py` skips these rows with a reason when the arrays are absent. |
| typed from `<file>` | The table is typed into the paper from the named shipped file. A test checks the typed values against that file where the brief for the table says so. |
| typed in the paper | The table is typed into the paper and has no shipped source beyond the configurations named in the row. |
| static; no generator recovered | The figure ships as an image. No script that produced it survives. |

`src/test_manifest.py` runs every command below from the repository root with `MPLBACKEND=Agg` and `SOURCE_DATE_EPOCH=0`, and checks that the tracked files are byte-identical afterwards.

## 1. Main text

Rows are in the order of the paper.

| Label | What it shows | Script | Artefact | Status |
|---|---|---|---|---|
| fig:pipeline (Figure 1) | The training pipeline | none | `figures/fig1_pipeline.pdf` | static; no generator recovered |
| tab:impl_details (Table 1) | Trainable parameters of the three configurations | none | `configs/layer_norm_gtl/`, `configs/lora_gtl/`, `configs/lora_gtl_gsd/` | typed in the paper |
| tab:indomain_results (Table 2) | In-domain FF++ Video-AUC, Ours and six baselines | Ours: `uv run python src/tables/reproduce_tables.py indomain`; baselines: `uv run python src/tables/emit_paper_tables.py` | `results/logs/main/`; `results/baselines/comparison_master.csv`, `results/baselines/tex/tables_table2_rows.tex` | regenerates from shipped inputs |
| tab:cross_results (Table 3) | Cross-dataset Video-AUC, 13 datasets and MEAN-17 | Ours: `uv run python src/tables/reproduce_tables.py cross`; baselines: `uv run python src/tables/emit_paper_tables.py` | `results/logs/main/`; `results/baselines/comparison_master.csv`, `results/baselines/tex/tables_table1_rows.tex` | regenerates from shipped inputs |
| tab:headline_deltas (Table 4) | MEAN-17 against GenD-DINO: delta, exact two-sided Wilcoxon p, wins | `uv run python src/tables/table4_wilcoxon.py` | `results/baselines/comparison_master.csv` | regenerates from shipped inputs |
| tab:training_stability (Table 5) | Peak epoch, peak AUC, post-peak slope and area under the curve, three seeds | `uv run python src/dynamics/table5.py` | `results/training_dynamics/val_curves.csv`; slope rule in `results/training_dynamics/README.md`; the curves are rebuilt from the logs by `uv run python src/dynamics/parse_train_logs.py` | regenerates from shipped inputs |
| fig:val_curves, fig:lora_dynamics, fig:training_dynamics, fig:advantage_and_dynamics (Figure 2) | Validation trajectory and cross-domain asymmetry by modality | `uv run python src/figures/make_figure2.py` | `figures/fig2_val_curves.pdf`, `.png`; reads TensorBoard events and test results under `results/logs/main/` | regenerates from shipped inputs |
| tab:ablation (Table 6) | Single-framework ablation | `uv run python src/tables/reproduce_tables.py ablations` | `results/logs/` | regenerates from shipped inputs |

## 2. Supplementary

Rows are in the order of the supplementary.

| Label | What it shows | Script | Artefact | Status |
|---|---|---|---|---|
| tab:supp_dataset_stats (Table S1) | Dataset statistics | none | none | typed in the paper |
| tab:supp_pub_vs_reeval_cross (Table S2) | Cross-dataset AUC, GenD-pipeline values against re-evaluation | `uv run python src/tables/emit_supplementary_tables.py` | `results/baselines/comparison_master.csv`, `results/published/published_numbers.csv`; `results/baselines/tex/supp_table_baseline_published_vs_reeval_cross.tex` | regenerates from shipped inputs |
| tab:supp_pub_vs_reeval_indomain (Table S3) | In-domain AUC, GenD-pipeline values against re-evaluation | `uv run python src/tables/emit_supplementary_tables.py` | as Table S2; `results/baselines/tex/supp_table_baseline_published_vs_reeval_indomain.tex` | regenerates from shipped inputs |
| tab:supp_orig_published_cross (Table S4) | Each baseline's original publication against the re-evaluation | none | `results/published/published_numbers.csv`, `results/baselines/comparison_master.csv`; checked by `src/tables/test_baselines.py` | typed from `results/published/published_numbers.csv` and `results/baselines/comparison_master.csv` |
| tab:supp_mean13_subset (Table S5) | The 13-dataset MEAN subset | Ours: `uv run python src/tables/reproduce_tables.py cross`; baselines: `uv run python src/tables/emit_paper_tables.py` | `results/logs/main/`; `results/baselines/comparison_master.csv` | regenerates from shipped inputs |
| fig:supp_heatmap (Figure S1) | Method by dataset heatmap | `uv run python src/figures/build_comparison_figures.py` | `figures/figS1_heatmap.pdf`, `.png`; reads `results/baselines/comparison_master.csv` | regenerates from shipped inputs |
| fig:supp_perdataset (Figure S2) | Per-dataset AUC for all nine methods | `uv run python src/figures/build_comparison_figures.py` | `figures/figS2_per_dataset.pdf`, `.png` | regenerates from shipped inputs |
| fig:supp_summary (Figure S3) | Summary across the three evaluation conditions | `uv run python src/figures/build_comparison_figures.py` | `figures/figS3_summary.pdf`, `.png` | regenerates from shipped inputs |
| fig:delta_gsd_vs_gend (Figure S4) | Per-dataset advantage of GSD-LoRA over GenD-DINO | `uv run python src/figures/build_comparison_figures.py` | `figures/figS4_delta_gsd_vs_gend.pdf`, `.png` | regenerates from shipped inputs |
| tab:supp_checkpoint_policy (Table S6) | MEAN-17 under each checkpoint-selection policy | `uv run python src/tables/reproduce_tables.py policy` | `results/logs/main/` | regenerates from shipped inputs |
| tab:k_sweep (Table S7) | Identity-subspace rank sweep on DeepSpeak-v2 | none | `results/repr_geometry/repr_geometry_summary.md`; the reading copy of the script is `src/repr_geometry/repr_geometry.py` | typed from `results/repr_geometry/repr_geometry_summary.md`; recomputation needs features (not shipped) |
| fig:centroid_map (Figure S5) | Real-to-fake displacement trajectories on DeepSpeak-v2 | none | `figures/figS5_centroid_map.png` | static; no generator recovered |
| tab:supp_arcface_pertype (Table S8) | ArcFace and z_perp similarity by manipulation-type combination | `uv run python src/arcface/correlate_across_pairs.py --arcface data/arcface/arcface_embeddings.npz --zperp data/arcface/zperp_embeddings.npz --out-md results/arcface/orthogonality_summary.md` | `results/arcface/orthogonality_summary.md`; the two arrays are release assets. The shipped `pair_pair_sample.csv` is the 10,000-row sample from the same seed; adding `--out-csv-sample <path>` rewrites it with cosines that differ in the sixth decimal (float32 rounding), so the command above leaves it alone | regenerates with release assets |
| tab:supp_arcface_idsep (Table S9) | Same-identity against different-identity separation | as Table S8 | `results/arcface/orthogonality_summary.md` | regenerates with release assets |
| tab:supp_loss_weights (Table S10) | Loss-weight inventory | none | values visible in `configs/layer_norm_gtl/`, `configs/lora_gtl/`, `configs/lora_gtl_gsd/` and `configs/ablations/sweeps/` | typed in the paper |
| tab:supp_ablation_sensitivity (Table S11) | Component-sensitivity ablations | `uv run python src/tables/reproduce_tables.py ablations` | `results/logs/ablations/` | regenerates from shipped inputs |
| tab:supp_hyperparam_sweeps (Table S12) | Single-seed hyperparameter sweeps | `uv run python src/tables/reproduce_tables.py sweeps` | `results/logs/ablations/` | regenerates from shipped inputs |
| fig:supp_box_summary (Figure S6) | Per-seed peak epoch and plateau width | `uv run python src/dynamics/build_training_dynamics_figure.py` | `figures/figS6_training_dynamics.pdf`, `.png`; reads `results/training_dynamics/val_curves.csv` and `overfitting_summary.csv` | regenerates from shipped inputs |
| tab:supp_full_breakdown_macro (Table S13) | Per-seed per-dataset AUC, macro-best checkpoint | `uv run python src/tables/reproduce_tables.py breakdown` | `results/logs/main/` | regenerates from shipped inputs |
| tab:supp_full_breakdown_dsv2 (Table S14) | As S13, DSv2-VAL selection | `uv run python src/tables/reproduce_tables.py breakdown` | `results/logs/main/` | regenerates from shipped inputs |
| tab:supp_full_breakdown_cdfv3 (Table S15) | As S13, CDFv3-VAL selection | `uv run python src/tables/reproduce_tables.py breakdown` | `results/logs/main/` | regenerates from shipped inputs |
| tab:supp_full_breakdown_ffsh (Table S16) | As S13, FF-FSH-VAL selection | `uv run python src/tables/reproduce_tables.py breakdown` | `results/logs/main/` | regenerates from shipped inputs |

## 3. Notes on what regenerates

- **Ours and baselines come from different files.** The Ours cells of Tables 2, 3 and S5 are computed from the run logs by `src/tables/reproduce_tables.py`. The baseline cells come from `results/baselines/comparison_master.csv` through the two emitters. The emitters also write Ours rows, but those are computed from `comparison_master.csv`, which averages per-seed values already rounded to two decimal places for some columns, so five Ours cells print 0.01 away from the paper there. The paper's values are the log-derived ones. `src/tables/test_baselines.py` checks that the Ours rows of `comparison_master.csv` match the logs to within that rounding.
- **Three exact ties.** Table 2 (GenD-CLIP, MEAN) and Table S3 (FA (ours), MEAN and MEAN delta) sit exactly on a half. The paper rounds half up. The emitters print with Python rounding, so they show the other neighbour. The tests compare the underlying values.
- **Table 4** is computed from `comparison_master.csv`: the exact two-sided Wilcoxon signed-rank test over the 17 datasets of MEAN-17.
- **Table 5** has no surviving generator. `src/dynamics/table5.py` was written for this repository; the slope rule is in `results/training_dynamics/README.md`.
- **Table S8**, the "all combinations" row, mean cos(z_perp): the printed value is the mean over the shipped 10,000-row sample (`results/arcface/pair_pair_sample.csv`). The full population gives 0.0540. Every other cell uses the full population.
- **Figures 1 and S5** ship as images. **Table S7** is typed from the summary file; recomputing it needs the extracted DeepSpeak-v2 features, which are not shipped.
- **Paired SLERP.** The configurations in `configs/` set `use_paired_slerp_aug: false`, which matches what ran. The run logs and the sidecar configurations of the released checkpoints keep the recorded `true`. The flag had no effect in the code that produced the numbers; see `method/METHOD.md`, `patches/slerp-restore.patch` and `results/slerp_rerun/`.
