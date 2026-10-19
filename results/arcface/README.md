# ArcFace identity check (supplementary Appendix G)

Tables S8 and S9 compare the OPA residual z_perp of the GSD-LoRA model with an independent identity space, ArcFace (InsightFace), on 1,820 real and fake frame pairs from FF++ (c23).

## Which script backs the tables

`src/arcface/correlate_across_pairs.py` backs Tables S8 and S9. It is the across-pair comparison: for every two pairs i and j it compares cos(z_perp_i, z_perp_j) with the ArcFace cosine of their real frames, and splits the result by manipulation type and by same or different identity. `MANIFEST.md` gives the command, which needs the two release assets in `data/arcface/`.

The other three scripts in `src/arcface/` are reading copies, kept to show how the inputs were made and what was tried first:

| Script | What it does | Reads | Writes | Runs from this repository |
|---|---|---|---|---|
| `rebuttal_arcface_embed.py` | ArcFace embedding of every frame in the manifest | `ff_pair_manifest.jsonl` and the FF++ frames | `arcface_embeddings.npz` (release asset) | No: it needs the FF++ frames and InsightFace. |
| `extract_zperp.py` | z_perp of every pair from a trained model; its usage example names the GSD-LoRA seed-42 checkpoint, epoch 7 | the checkpoint, its config, the manifest and the FF++ frames | `zperp_embeddings.npz` (release asset) | No: it needs the full checkpoint, the training code and the frames. |
| `rebuttal_arcface_correlate.py` | The first, within-pair comparison: the ArcFace and z_perp cosines between the real and the fake frame of each pair | the two release assets | `pair_similarities.csv` and a Markdown summary | Yes, with the release assets. It backs no table. |

The within-pair comparison is degenerate: OPA gives the real and the fake frame of a pair the same residual, so the z_perp cosine is 1 in every row of `pair_similarities.csv` (to float32 rounding). The across-pair comparison replaced it.

## Files here

- `ff_pair_manifest.jsonl`: the 3,640 frames (1,820 pairs) read by both embedding scripts, with frame paths relative to the FF++ root.
- `orthogonality_summary.md`: Tables S8 and S9, written by `correlate_across_pairs.py`.
- `pair_pair_sample.csv`: 10,000 of the 1,655,290 pair-pair combinations, the sample that `correlate_across_pairs.py --out-csv-sample` draws with its default seed 0 (a rerun gives cosines that differ in the sixth decimal; see the Table S8 row of `MANIFEST.md`). The "all combinations" mean of cos(z_perp) in Table S8 is the mean over this sample.
- `pair_similarities.csv`: the within-pair table, one row per pair. Run on the two release assets, `rebuttal_arcface_correlate.py` gives the same two cosines in all 1,820 rows. Its `type` column differs: the shipped file says `swap` or `reenactment`, while the reading copy writes `unknown` for every row, because its keyword list holds short tags (`DF`, `F2F`, `NT`, `FS_`) and the manifest uses the full FF++ method names.
