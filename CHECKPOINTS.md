# CHECKPOINTS

The nine headline models of the paper (LN, LoRA and GSD-LoRA, each at seeds 42, 137 and 1729), released as the parts that training changed. Each file is a `.safetensors` archive with a JSON sidecar of the same name in `checkpoints/` that records how it was made.

## The files

Each file is made from the checkpoint the paper evaluated: the one named on the first line ("Testing checkpoint: ...") of the run's macro-best `test.log`, `results/logs/main/<run>/test/epochNN_best_macro/test.log`. The epoch and validation Video-AUC are those in that checkpoint's file name. `src/checkpoints/test_evaluated_checkpoints.py` checks every sidecar against its `test.log`. The files are assets of the `v1.0` release of this repository and are not kept in git. The sha256 of each is also recorded as `parts_sha256` in its sidecar.

| Run | Epoch | Val AUC | File | Size | sha256 | Release URL |
|---|---|---|---|---|---|---|
| LN s42 | 18 | 0.8529 | `orthoflow_gtl_ln_s42.safetensors` | 411 KiB (421,096 B) | `76845d7649ae7828a6340bfd3b667affe6fac347ccf4e26773975f49806c0654` | [orthoflow_gtl_ln_s42.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_ln_s42.safetensors) |
| LN s137 | 24 | 0.8555 | `orthoflow_gtl_ln_s137.safetensors` | 411 KiB (421,096 B) | `c4915518ba74066147c215aa056920710b3ef038c24cf81d672ea97a0b4a3ffa` | [orthoflow_gtl_ln_s137.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_ln_s137.safetensors) |
| LN s1729 | 9 | 0.8527 | `orthoflow_gtl_ln_s1729.safetensors` | 411 KiB (421,096 B) | `211ff078d9d87e653cb0df005bfbc5fa1d917be03ac32fe58b3a9cf995a6eba9` | [orthoflow_gtl_ln_s1729.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_ln_s1729.safetensors) |
| LoRA s42 | 4 | 0.8717 | `orthoflow_gtl_lora_s42.safetensors` | 116.1 MiB (121,707,456 B) | `7aef0cb87d651a27f8f58bdc380807967c944a6b58b8fcfdda1a9c758333b4f7` | [orthoflow_gtl_lora_s42.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_lora_s42.safetensors) |
| LoRA s137 | 5 | 0.8502 | `orthoflow_gtl_lora_s137.safetensors` | 116.1 MiB (121,707,456 B) | `643635bf84e8e151b573cd00310a0064b66b7c9e12321ae7636ee55e860c931c` | [orthoflow_gtl_lora_s137.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_lora_s137.safetensors) |
| LoRA s1729 | 1 | 0.8602 | `orthoflow_gtl_lora_s1729.safetensors` | 116.1 MiB (121,707,456 B) | `55a9891770ee418447ce79e88cf2c9e2059912c5e442be71cb7278673b4f346b` | [orthoflow_gtl_lora_s1729.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_lora_s1729.safetensors) |
| GSD-LoRA s42 | 7 | 0.8678 | `orthoflow_gtl_gsd_lora_s42.safetensors` | 116.1 MiB (121,707,456 B) | `684b9314d3ba7d7e8af796a7ee6101c9cf56072c2af3b92234824d4dc2af1587` | [orthoflow_gtl_gsd_lora_s42.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_gsd_lora_s42.safetensors) |
| GSD-LoRA s137 | 3 | 0.8697 | `orthoflow_gtl_gsd_lora_s137.safetensors` | 116.1 MiB (121,707,456 B) | `7ea46183beeba8fb5f33c61293fd656f05215cab5c2f8f5d3fbe1684703cd7fd` | [orthoflow_gtl_gsd_lora_s137.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_gsd_lora_s137.safetensors) |
| GSD-LoRA s1729 | 7 | 0.8651 | `orthoflow_gtl_gsd_lora_s1729.safetensors` | 116.1 MiB (121,707,456 B) | `96903341644799ad2c2261b0386051b8018da1a8864fd874514ef97e163959f6` | [orthoflow_gtl_gsd_lora_s1729.safetensors](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/orthoflow_gtl_gsd_lora_s1729.safetensors) |

Two runs need a note:

- **LN s137.** The evaluated epoch 24 (val 0.8555) is the file named in the test log, although epoch 23 logged a marginally higher macro validation score (0.8556).
- **GSD-LoRA s1729.** The test log names `best_CDFv3-VAL.ckpt`, the file kept by the CDFv3-VAL checkpoint callback, not a file named by epoch. Its `state_dict` equals that of `epoch=07-video_auc=0.8651.ckpt` tensor by tensor (all 724 tensors; both record epoch 7 and global step 5760), so the release is extracted from the epoch-7 file. The sidecar records the alias in `evaluated_as`, with the alias file's own sha256 (`29b5349a85f3da8195dd471e9d8900362ab83892d2e42ba32cb816035e71097f`), because the two files are not byte-identical.

Download a file into `data/release/` and check it with `sha256sum`. This repository is private until it is made public, so downloads need repository access until then.

## What was removed

In each original Lightning checkpoint almost all of the weight is Meta's frozen DINOv3 ViT-L/16 backbone, which is not ours to redistribute, plus training state. A released file keeps only

- every backbone tensor that differs from the base weights (the 98 tuned LayerNorm tensors of the LN models),
- the LoRA adapter tensors `lora_A` and `lora_B` (288 tensors, for the LoRA and GSD-LoRA models),
- the classification head (2 tensors for LN, a linear head; 4 for LoRA and GSD-LoRA, a two-layer head),

under the key names of the original `state_dict`.

Removed:

- the frozen backbone: 317 tensors for LN and 415 for LoRA and GSD-LoRA, each bit-identical to the base weights;
- the training objective's own parameters (`loss_fn.*`, 17 tensors), which inference does not call;
- optimiser state, learning-rate schedulers, loop state, step counters and callback state, the last of which recorded absolute paths of the training machine.

The extraction script is `src/checkpoints/extract_trained_parts.py`. Its `verify` function rebuilds the model from the released parts and the base weights and compares the result with the original checkpoint tensor by tensor; all nine files gave no mismatches. For the three seed-42 models, the released parts plus the base weights also give bitwise identical logits to the original checkpoint on a fixed random input on CPU.

## How to reassemble a model

1. Fetch the base weights from Hugging Face: `facebook/dinov3-vitl16-pretrain-lvd1689m`, revision `ea8dc2863c51be0a264bab82070e3e8836b02d51` (gated; accept Meta's terms first). The sidecar field `hf_sha256` is the sha256 of that revision's `model.safetensors`.
2. Build the detector from the sidecar's `config` field, which is the resolved training configuration as recorded in the checkpoint, with paths removed. It records `use_paired_slerp_aug: true`, a flag with no effect in that code: the paired SLERP step had been removed, so it did not run (`CORRECTIONS.md`, `method/METHOD.md`).
3. Load the released tensors over the base weights. The keys in the file are the original names:
   - backbone tensors are `model._orig_mod.backbone.model.<hugging-face key>`; `_orig_mod` comes from `torch.compile`, and the sidecar field `key_prefix` gives the full prefix;
   - for LoRA and GSD-LoRA the backbone is wrapped by PEFT (`peft_wrapped: true`). Its keys gain `base_model.model.`, and each adapted linear layer keeps its base weight as `<name>.base_layer.weight` and `.bias`. The adapter is rank 64, alpha 128, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj` and `down_proj`;
   - the head is `model._orig_mod.head.<name>`.
4. For LN, the tuned LayerNorm tensors replace those of the base weights. For LoRA and GSD-LoRA, wrap the base model with PEFT first, then load the adapter tensors and the head.

`reassemble` in `src/checkpoints/extract_trained_parts.py` does this at the level of `state_dict` entries and is the reference for the key mapping. The sidecar `kept_keys` and `dropped_groups` fields count what each file holds and what was left out. The optional `evaluated_as` field appears only for GSD-LoRA s1729 (see above). The method code as run is in `method/`.

Running a model on your own video is done by the `deepfake-workbench` adapter (dfwb-research), which loads these files. It is not yet released.

## ArcFace arrays

Tables S8 and S9 regenerate from two arrays that are too large for git. They are release assets of `v1.0`; download them into `data/arcface/`.

| File | Size | sha256 | Release URL |
|---|---|---|---|
| `arcface_embeddings.npz` | 8,085,826 B | `0cd62084c0007883cc586b30e159eb66e3dc9ac33ac9c9948f10db71d1be4de8` | [arcface_embeddings.npz](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/arcface_embeddings.npz) |
| `zperp_embeddings.npz` | 15,540,538 B | `859375ffd996db459b108c008ae4e151eaf8798f27e825d3d35419d9f07d564c` | [zperp_embeddings.npz](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/zperp_embeddings.npz) |

## Licence

The released model files are derivatives of DINOv3 and are distributed under the DINOv3 License, in `LICENSE-DINOv3.md` ([release copy](https://github.com/lukegcollins/BMVC26-PGS/releases/download/v1.0/LICENSE-DINOv3.md)). The base weights come from Hugging Face under the same licence and are not redistributed here. The code and text of this repository are MIT (`LICENSE`).
