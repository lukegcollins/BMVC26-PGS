# Method code as run

This directory holds a read-only copy of the files that define the method, exactly as they were when the
paper's headline models were trained. The files under `method/src/` are byte-identical to the same paths in the
DeepfakeWorkbench repository at commit `2813161f2da9c2538b8c913b5182909dc658972b`. None of them has been edited,
reformatted or cleaned up.

The code is not installed or run here. Ruff and pytest skip `method/`, and the repository environment does not
provide its dependencies. Running a released model on new videos is done by the dfwb adapter
(`deepfake-workbench`, in `dfwb-research`), which loads the checkpoints released with this repository.

## Which commit, and why

- The nine headline runs (LN, LoRA and GSD-LoRA, each with seeds 42, 137 and 1729) recorded git commit
  `306d567318e1453a93ccac2caa625fc6bb8b4381`. That commit was made in a second clone of the Workbench
  repository.
- Commit `2813161f2da9c2538b8c913b5182909dc658972b` has the same tree,
  `4004234a01d5169086fea1e06323871289cace43`. The two commits therefore hold identical files, and this copy is
  the code the headline runs used.
- The run logs under `results/logs/` carry the `git_commit` field anonymised. The commit above comes from the
  original, unanonymised records.

With a clone of the Workbench repository, the byte check runs as

```
PGS_WORKBENCH=<path to the Workbench clone> uv run pytest src/test_method.py
```

and compares every file here with `git show 2813161f:<path>`. Without that variable the check is skipped.

## What is included

The files were chosen by following imports at `2813161f`, starting from the detector, the DINOv3 backbone, the
heads, the configuration, the transforms and the training objectives, and adding every module of the
repository that those files import unconditionally.

- `src/detectors/`: `orthoflow_detector.py`, `base_detector.py`, `registry.py`.
- `src/backbones/`: `dinov3_backbone.py` (Hugging Face `facebook/dinov3-vitl16-pretrain-lvd1689m`, with
  LayerNorm tuning or LoRA through PEFT), `base.py`, `registry.py`, and `dinov2_backbone.py`, which the
  registry imports unconditionally. The runs do not use DINOv2.
- `src/core/registry.py` and `src/heads/`.
- `src/configs/`: `base_config.py`, `__init__.py`.
- `src/dataset/transforms/`: `builder.py`, `base_strategy.py`, `paired_augmentation.py`.
- `src/dataset/`: `factory.py`, `core/base.py` and `components/{single,paired,dual_paired}.py`, which the
  package's `__init__.py` imports.
- `src/loss/`: the OrthoPair-GTL objective (`_orthoflow_gtl_core.py`, `orthoflow_gtl.py`), the GSD set loss
  and its combination with OrthoPair-GTL (`gsd_set_loss.py`, `orthoflow_gtl_gsd.py`), `abstract_loss.py` and
  `registry.py`. Every shipped configuration selects `loss_type: orthoflow_gtl_gsd`, which wraps the
  OrthoPair-GTL objective.
- The `__init__.py` of each package above.

Points to note:

- `src/dataset/transforms/paired_augmentation.py` is truncated at this commit: it ends inside a docstring, so
  Python cannot import it. No file at the commit imports it, so it played no part in the runs. It is kept byte
  for byte like everything else.
- The four registries try to import optional modules inside `try ... except ImportError` blocks. Those are
  other methods' backbones, detectors and losses. Most of them do not exist at this commit; the one that does,
  `backbones/xclip_backbone.py`, is unrelated to the method. None of them is included.
- The training loop (`lightning/`), the command-line entry points (`cli/`), the samplers, the dataset pairing
  rules, the metrics and the utilities are not imported by the files above and are not included.

## The ablation runs used a later commit

The ablation and sensitivity-sweep runs under `results/logs/ablations/` (25 ablation runs and 10 sweep runs)
recorded commit `e498a86797562168335328d5c74b08e986521d57`, eleven commits after `2813161f`. That code is not
copied here. Between the two commits, `src/` changes in 29 files (4,367 lines added, 108 removed).

The detector, the DINOv3 backbone, the heads, the transforms, the loss registry and the OrthoPair-GTL objective
(`_orthoflow_gtl_core.py`, `orthoflow_gtl.py`, `orthoflow_gtl_gsd.py`) are the same at both commits. The changes
that touch the method are these.

- `loss/gsd_set_loss.py` gains opt-in "v2" Fisher-style terms: method-grouped concentration, coherence measured
  on unnormalised residual centroids, and per-class dispersion. Each sits behind a flag that is off by default.
  The original per-video loop was restructured around them. With the flags off the original terms are
  expected to be unchanged, but no test or rerun has shown this.
- `configs/base_config.py` gains two temporal-sampling settings, `start_offset` (default 0) and
  `clip_step_mode` (default `"position"`).
- `dataset/components/single.py` changes how `StandardSequenceDataset` builds clips. By default it now steps
  through the frames' positions in the folder, where `2813161f` stepped through the integers parsed from the
  frame file names. That class is built only when `sequence_length` is greater than 1. Every shipped
  configuration sets `sequence_length: 1`, so the runs use the frame datasets and this change does not reach
  them. `dataset/components/paired.py` accepts the two new settings without using them.
- `lightning/module.py` adds a SupCon branch to the loss builder, writes per-video validation predictions to
  CSV, and names each validation and test loader from its dataset entry in the configuration instead of from
  per-sample metadata. `lightning/datamodule.py` stops keeping test-loader workers alive between test sets.
  `cli/test.py` writes each test set's outputs as soon as that set finishes and frees memory before the next.
- `backbones/registry.py` resolves a backbone name by its longest registered prefix. `dinov3_vitl16`, the name
  every configuration uses, resolves to the same DINOv3 class as before.
- New modules that no shipped configuration selects: `loss/classification.py` and `loss/supcon.py`.
- New modules unrelated to this method: V-JEPA 2 and V-JEPA 2.1 backbones (with vendored V-JEPA 2.1 code), a
  hybrid backbone, adapters for the comparison detectors and their registration, and an extra name for the
  KoDF pairing rule.

## Paired SLERP: flag accepted, inactive

At both `2813161f` and `e498a86`, the `use_paired_slerp_aug` setting is read from the configuration, passed to
the objective, stored, and written to the training log. No code reads the stored value, so no SLERP points were
added to the uniformity term in any reported run. The training logs print `Use SLERP aug: True` for runs whose
configuration set the flag; it had no effect. The configurations under `configs/` set it to `false` so that
they describe what actually ran.

`patches/slerp-restore.patch` restores the augmentation on top of the frozen file. From the repository root:

```
git apply --check patches/slerp-restore.patch
git apply patches/slerp-restore.patch
```

It changes only `method/src/loss/_orthoflow_gtl_core.py`:

- it adds `_slerp_augmentation`, which draws `tau ~ U(0, 1)` for each aligned real and fake pair (once per
  `slerp_pair_multiplier`) and returns the normalised point at `tau` on the great-circle arc between the two
  embeddings;
- when `use_paired_slerp_aug` is true, the module is training and the batch holds at least one pair, those
  points are appended to the embeddings in the uniformity term;
- the log line also reports the multiplier.

Its added and removed lines are identical to the change Workbench commit
`95d7767a3a848c8d83e73650e43a46c8116eea9e` made to the same file. Only the surrounding context differs,
because the frozen file predates later, unrelated edits. With the patch applied and the shipped configurations,
which set the flag to `false`, behaviour is unchanged.

A registered three-seed rerun trained each headline arm with the restored SLERP on and off and compared the
mean video AUC over the 17 test sets (MEAN-17). The matched on minus off differences are:

| Arm | On minus off, MEAN-17 (pp) |
|---|---|
| LN | +0.13 |
| LoRA | -0.46 |
| GSD-LoRA | -0.49 |

The rerun read its frames from a different extraction from the paper's runs (amendment A1 of its
registration), so its absolute AUCs are not the paper's; only the matched differences compare. The
registration, the verdict and the per-run scores are in `results/slerp_rerun/`.
