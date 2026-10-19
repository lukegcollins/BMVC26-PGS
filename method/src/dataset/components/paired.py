"""
Paired dataset components for contrastive deepfake detection training.

Both PairedFrameDataset and PairedSequenceDataset match each fake video to
its original real video using the pair_id resolved from the JSONL manifest
(or the legacy pair_id field).

PairedSequenceDataset supports three temporal alignment modes to handle
re-encoding drift between real and fake frame directories:

  strict_intersect
      Compute the mathematical intersection of available frame filenames
      between the real and fake directories.  Only load frames whose filename
      integer (e.g. 00032 from "00032.png") exists in BOTH directories.
      Returns None (skips the sample) if any required frame is missing.

  tolerance
      Use a two-pointer scan to pair fake frames to the closest real frame
      whose index is within ±temporal_alignment_tolerance.  Real frames are
      consumed in order to prevent temporal regression.  Returns None if any
      fake clip frame cannot be matched within tolerance.

  index_aligned  (default / Smart Mode)
      Compare total frame counts.  If |real_frames| == |fake_frames| (e.g.
      both have exactly 32 after fixed_num_frames extraction), zip them by
      sorted position (0th↔0th, 1st↔1st, …) regardless of the actual
      filename integer.  This gracefully handles the common case where
      re-encoding shifts every frame index by a small constant offset.
      Falls back to strict_intersect when counts differ.
"""

import logging
import random
from typing import Literal

import torch

from src.dataset.core.base import DeepfakeBaseDataset

logger = logging.getLogger(__name__)

TemporalAlignmentMode = Literal["strict_intersect", "tolerance", "index_aligned"]


class PairedFrameDataset(DeepfakeBaseDataset):
    """
    Paired dataset for single-frame spatial contrastive learning.

    Iterates over fake frames and finds the temporally-aligned real frame
    using the configured temporal_alignment_mode.

    Returns:
        {
            "real":  Tensor (C, H, W),
            "fake":  Tensor (C, H, W),
            "label": int (always 1),
            "meta":  dict,
        }
    """

    def __init__(
        self,
        *args,
        pairing_mode: str = "strict",
        sync_augmentations: bool = True,
        sync_threshold: int = 5,
        temporal_alignment_mode: TemporalAlignmentMode = "index_aligned",
        temporal_alignment_tolerance: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pairing_mode = pairing_mode
        self.sync_augment = sync_augmentations
        self.sync_threshold = sync_threshold
        self.temporal_alignment_mode = temporal_alignment_mode
        self.temporal_alignment_tolerance = temporal_alignment_tolerance

        # ── Fake anchor index ─────────────────────────────────────────────────
        self.fakes = [s for s in self.samples if s["label"] == 1]

        # ── Real lookup: video_id → {frame_index → frame_path} ───────────────
        self.real_lookup: dict[str, dict[int, str]] = {}
        self.real_paths_map: dict[str, str] = {}  # video_id → video_path

        for s in self.samples:
            if s["label"] == 0:
                vid = s["video_id"]
                idx = self._extract_frame_index(s["frame_name"])
                if vid not in self.real_lookup:
                    self.real_lookup[vid] = {}
                    self.real_paths_map[vid] = s["video_path"]
                self.real_lookup[vid][idx] = s["frame_path"]

        # For index_aligned mode: store sorted frame lists per real video
        self.real_sorted_frames: dict[str, list[int]] = {
            vid: sorted(frame_map.keys())
            for vid, frame_map in self.real_lookup.items()
        }
        self.real_video_ids = list(self.real_lookup.keys())

    def __len__(self):
        return len(self.fakes)

    def __getitem__(self, idx):
        fake_meta = self.fakes[idx]

        if self.pairing_mode == "random_source":
            real_vid_id = random.choice(self.real_video_ids)
            real_idx = random.choice(list(self.real_lookup[real_vid_id].keys()))
            real_path = self.real_lookup[real_vid_id][real_idx]
        else:
            real_vid_id = fake_meta.get("pair_id")
            real_path = self._find_aligned_real_frame(fake_meta, real_vid_id)

        if real_path is None:
            # Fallback: random real frame to avoid None propagation
            if self.real_video_ids:
                real_vid_id = random.choice(self.real_video_ids)
                real_path = self.real_lookup[real_vid_id][
                    random.choice(list(self.real_lookup[real_vid_id].keys()))
                ]
            else:
                real_path = fake_meta["frame_path"]

        fake_img = self._load_image(fake_meta["frame_path"])
        real_img = self._load_image(real_path)

        if self.transform:
            if self.sync_augment:
                res = self.transform(image=fake_img, image_pair=real_img)
                fake_img, real_img = res["image"], res["image_pair"]
            else:
                fake_img = self.transform(image=fake_img)["image"]
                real_img = self.transform(image=real_img)["image"]

        return {
            "real": real_img,
            "fake": fake_img,
            "label": 1,
            "meta": {
                "task": fake_meta["task"],
                "fake_video_id": fake_meta["video_id"],
                "real_video_id": real_vid_id,
                "video_id": fake_meta.get("jsonl_id") or fake_meta["video_id"],
                "dataset_name": fake_meta.get("dataset_name", fake_meta["task"]),
            },
        }

    def _find_aligned_real_frame(self, fake_meta: dict, real_vid_id: str | None) -> str | None:
        """Find the real frame path aligned to this fake frame."""
        if not real_vid_id or real_vid_id not in self.real_lookup:
            return None

        fake_frame_index = self._extract_frame_index(fake_meta["frame_name"])
        real_frame_map = self.real_lookup[real_vid_id]
        real_sorted = self.real_sorted_frames[real_vid_id]

        mode = self.temporal_alignment_mode

        if mode == "strict_intersect":
            return real_frame_map.get(fake_frame_index)

        elif mode == "tolerance":
            tol = self.temporal_alignment_tolerance
            for ri in real_sorted:
                if abs(ri - fake_frame_index) <= tol:
                    return real_frame_map[ri]
            return None

        elif mode == "index_aligned":
            # Build sorted fake index list for this video
            fake_sorted = sorted(
                self._extract_frame_index(s["frame_name"])
                for s in self.samples
                if s["label"] == 1 and s["video_id"] == fake_meta["video_id"]
                   and s["task"] == fake_meta["task"]
            )
            if len(fake_sorted) == len(real_sorted):
                try:
                    pos = fake_sorted.index(fake_frame_index)
                    return real_frame_map[real_sorted[pos]]
                except (ValueError, IndexError):
                    pass
            # Fallback to strict_intersect when counts differ
            return real_frame_map.get(fake_frame_index)

        # Unknown mode: nearest neighbour fallback
        closest = min(real_sorted, key=lambda x: abs(x - fake_frame_index))
        return real_frame_map[closest]


class PairedSequenceDataset(DeepfakeBaseDataset):
    """
    Paired sequence dataset for spatio-temporal contrastive learning.

    For each fake video clip anchor, finds the temporally-aligned clip from
    the paired real video using one of three alignment modes.

    Temporal alignment modes
    ------------------------
    strict_intersect
        Only include frames whose filename integer exists in both real and
        fake directories.  Returns None (collate_fn drops the sample) if any
        clip frame is missing in the real video.

    tolerance
        Match each fake frame to the nearest real frame within
        ±temporal_alignment_tolerance, using a two-pointer scan to prevent
        temporal regression.  Returns None if any frame cannot be matched.

    index_aligned  (default)
        If the real and fake videos have the same total number of extracted
        frames (typical when both were preprocessed with fixed_num_frames),
        map fake clip position i → real_frames_sorted[i] regardless of
        filename integer value.  Falls back to strict_intersect if counts
        differ.

    Returns:
        {
            "real":  Tensor (T, C, H, W),
            "fake":  Tensor (T, C, H, W),
            "label": int (always 1),
            "meta":  dict,
        }
        or None (sample dropped by collate_fn) on alignment failure.
    """

    def __init__(
        self,
        *args,
        sequence_length: int = 16,
        temporal_stride: int = 1,
        window_step: int = 8,
        max_sequences_per_video: int = 10,
        pairing_mode: str = "strict",
        sync_augmentations: bool = True,
        sync_threshold: int = 5,
        aggregation_method: str = "mean",
        aggregation_threshold: float = 0.5,
        temporal_alignment_mode: TemporalAlignmentMode = "index_aligned",
        temporal_alignment_tolerance: int = 2,
        **kwargs,
    ):
        kwargs["scan_mode"] = "all"
        super().__init__(*args, **kwargs)

        self.seq_len = sequence_length
        self.temp_stride = temporal_stride
        self.win_step = window_step
        self.max_seq = max_sequences_per_video
        self.pairing_mode = pairing_mode
        self.sync_augment = sync_augmentations
        self.sync_threshold = sync_threshold
        self.aggregation_method = aggregation_method
        self.aggregation_threshold = aggregation_threshold
        self.temporal_alignment_mode: TemporalAlignmentMode = temporal_alignment_mode
        self.temporal_alignment_tolerance = temporal_alignment_tolerance

        # ── Group fake frames by (task, video_id) ─────────────────────────────
        # FF++ has the same video_id in multiple tasks — must key on both.
        self.fake_video_map: dict[tuple, list[dict]] = {}
        for s in self.samples:
            if s["label"] == 1:
                vid_key = (s.get("task", "unknown"), s["video_id"])
                if vid_key not in self.fake_video_map:
                    self.fake_video_map[vid_key] = []
                self.fake_video_map[vid_key].append(s)

        # ── Group real frames by video_id ─────────────────────────────────────
        self.real_video_map: dict[str, list[dict]] = {}
        self.real_paths_map: dict[str, str] = {}
        for s in self.samples:
            if s["label"] == 0:
                if s["video_id"] not in self.real_video_map:
                    self.real_video_map[s["video_id"]] = []
                    self.real_paths_map[s["video_id"]] = s["video_path"]
                self.real_video_map[s["video_id"]].append(s)

        # ── Sort real frames and cache total counts ────────────────────────────
        for vid in self.real_video_map:
            self.real_video_map[vid].sort(
                key=lambda x: self._extract_frame_index(x["frame_name"])
            )
        self._real_frame_counts: dict[str, int] = {
            vid: len(frames) for vid, frames in self.real_video_map.items()
        }

        # ── Build fake anchor clips ───────────────────────────────────────────
        # Each anchor is a tuple: (clip_frames, start_pos_in_video, vid_key)
        # start_pos is the absolute index of clip_frames[0] in its sorted video
        # frame list — used by index_aligned mode to select the same position
        # in the real video.
        self.anchors: list[tuple[list[dict], int, tuple]] = []
        self._fake_frame_counts: dict[tuple, int] = {}

        for vid_key, frames in self.fake_video_map.items():
            frames.sort(key=lambda x: self._extract_frame_index(x["frame_name"]))
            self._fake_frame_counts[vid_key] = len(frames)

            span = (self.seq_len - 1) * self.temp_stride + 1
            curr: list[tuple[list[dict], int]] = []

            if len(frames) < span:
                if len(frames) > self.seq_len:
                    curr.append((frames[: self.seq_len], 0))
                else:
                    curr.append((frames, 0))
            else:
                for i in range(0, len(frames) - span + 1, self.seq_len):
                    idxs = [i + k * self.temp_stride for k in range(self.seq_len)]
                    curr.append(([frames[x] for x in idxs], i))

            if self.max_seq and len(curr) > self.max_seq:
                curr = random.sample(curr, self.max_seq)

            for clip, start_pos in curr:
                self.anchors.append((clip, start_pos, vid_key))

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        fake_clip, start_pos, vid_key = self.anchors[idx]

        # Edge padding for short clips
        if len(fake_clip) < self.seq_len:
            pad = self.seq_len - len(fake_clip)
            fake_clip = fake_clip + [fake_clip[-1]] * pad

        # ── Find paired real video ────────────────────────────────────────────
        real_vid_id = fake_clip[0].get("pair_id")
        if not real_vid_id or real_vid_id not in self.real_video_map:
            return None  # Pair not found — collate_fn drops this sample

        real_frames_sorted = self.real_video_map[real_vid_id]

        # ── Temporal alignment ────────────────────────────────────────────────
        real_clip = self._align_clip(
            fake_clip=fake_clip,
            start_pos=start_pos,
            vid_key=vid_key,
            real_frames_sorted=real_frames_sorted,
        )
        if real_clip is None:
            return None

        # Pad real clip if needed (only in non-strict paths)
        if len(real_clip) < self.seq_len:
            if not real_clip:
                return None
            pad = self.seq_len - len(real_clip)
            real_clip = real_clip + [real_clip[-1]] * pad

        # ── Load & transform with consistent replay ───────────────────────────
        fake_tensors = []
        real_tensors = []
        replay_params = None

        fake_img_0 = self._load_image(fake_clip[0]["frame_path"])
        if self.transform:
            res = self.transform(image=fake_img_0)
            fake_tensors.append(res["image"])
            replay_params = res.get("replay")

            for item in fake_clip[1:]:
                img = self._load_image(item["frame_path"])
                res = self.transform.replay(replay_params, image=img) if replay_params else self.transform(image=img)
                fake_tensors.append(res["image"])

            for item in real_clip:
                img = self._load_image(item["frame_path"])
                res = self.transform.replay(replay_params, image=img) if replay_params else self.transform(image=img)
                real_tensors.append(res["image"])
        else:
            fake_tensors.append(torch.from_numpy(fake_img_0).permute(2, 0, 1))
            for item in fake_clip[1:]:
                img = self._load_image(item["frame_path"])
                fake_tensors.append(torch.from_numpy(img).permute(2, 0, 1))
            for item in real_clip:
                img = self._load_image(item["frame_path"])
                real_tensors.append(torch.from_numpy(img).permute(2, 0, 1))

        return {
            "real": torch.stack(real_tensors),   # (T, C, H, W)
            "fake": torch.stack(fake_tensors),   # (T, C, H, W)
            "label": 1,
            "meta": {
                "task": fake_clip[0]["task"],
                "fake_video_id": fake_clip[0]["video_id"],
                "real_video_id": real_vid_id,
                "video_id": fake_clip[0]["video_id"],
                "dataset_name": fake_clip[0].get("dataset_name", fake_clip[0]["task"]),
                "sequence_start_frame": self._extract_frame_index(fake_clip[0]["frame_name"]),
                "sequence_length": self.seq_len,
                "temporal_stride": self.temp_stride,
                "temporal_alignment_mode": self.temporal_alignment_mode,
            },
        }

    # =========================================================================
    # Temporal alignment strategies
    # =========================================================================

    def _align_clip(
        self,
        fake_clip: list[dict],
        start_pos: int,
        vid_key: tuple,
        real_frames_sorted: list[dict],
    ) -> list[dict] | None:
        """
        Dispatch to the configured temporal alignment strategy.

        Returns a list of seq_len real frame entries, or None on failure.
        """
        mode = self.temporal_alignment_mode
        if mode == "strict_intersect":
            return self._align_strict_intersect(fake_clip, real_frames_sorted)
        elif mode == "tolerance":
            return self._align_tolerance(fake_clip, real_frames_sorted)
        elif mode == "index_aligned":
            return self._align_index_aligned(fake_clip, start_pos, vid_key, real_frames_sorted)
        else:
            logger.warning(
                f"Unknown temporal_alignment_mode '{mode}', "
                f"falling back to index_aligned."
            )
            return self._align_index_aligned(fake_clip, start_pos, vid_key, real_frames_sorted)

    def _align_strict_intersect(
        self,
        fake_clip: list[dict],
        real_frames_sorted: list[dict],
    ) -> list[dict] | None:
        """
        Strict intersection: every fake frame filename must exist in the real
        directory.  Returns None if ANY frame is missing.

        This is the most conservative mode.  It preserves exact temporal
        correspondence but may produce very few or zero valid pairs when
        re-encoding causes systematic frame index shifts.
        """
        real_by_name: dict[str, dict] = {
            f["frame_name"]: f for f in real_frames_sorted
        }
        real_clip: list[dict] = []
        for ff in fake_clip:
            rf = real_by_name.get(ff["frame_name"])
            if rf is None:
                return None  # Strict: any missing = drop sample
            real_clip.append(rf)
        return real_clip

    def _align_tolerance(
        self,
        fake_clip: list[dict],
        real_frames_sorted: list[dict],
    ) -> list[dict] | None:
        """
        Tolerance matching: pair each fake frame to the nearest real frame
        whose index is within ±tolerance.

        Two-pointer scan prevents temporal regression (each selected real
        frame must come after the previously selected one).  Returns None if
        any fake clip frame cannot be matched within tolerance.

        Example (tolerance=2):
            Fake indices:  [00, 04, 08, 12, 16, 20, 24, 28]
            Real indices:  [02, 06, 10, 14, 18, 22, 26, 30]
            Pairs:         (00↔02), (04↔06), (08↔10), …  (all within tol=2)
        """
        tol = self.temporal_alignment_tolerance
        # Build sorted (index, frame_entry) list for real frames
        real_indexed = [
            (self._extract_frame_index(f["frame_name"]), f)
            for f in real_frames_sorted
        ]

        real_clip: list[dict] = []
        real_pos = 0  # cursor: never regress

        for ff in fake_clip:
            fi = self._extract_frame_index(ff["frame_name"])
            matched: dict | None = None

            # Scan forward from real_pos to find closest within tolerance
            for pos in range(real_pos, len(real_indexed)):
                ri, rf = real_indexed[pos]
                if abs(ri - fi) <= tol:
                    matched = rf
                    real_pos = pos + 1  # advance cursor past this match
                    break
                if ri > fi + tol:
                    break  # Past the tolerance window; no match possible

            if matched is None:
                return None  # Cannot match within tolerance → drop sample
            real_clip.append(matched)

        return real_clip

    def _align_index_aligned(
        self,
        fake_clip: list[dict],
        start_pos: int,
        vid_key: tuple,
        real_frames_sorted: list[dict],
    ) -> list[dict] | None:
        """
        Index-aligned (Smart Mode): map clip frames by sorted-list position.

        If the total number of extracted frames is identical for real and fake
        (the common case when both are processed with fixed_num_frames=32),
        the 0th fake frame corresponds to the 0th real frame in chronological
        order, the 1st to the 1st, and so on — completely ignoring the actual
        filename integers.

        This gracefully handles systematic per-frame drift caused by
        re-encoding (e.g. real=00000, 00004, … vs fake=00001, 00005, …).

        Falls back to strict_intersect when frame counts differ.

        Example (fixed_num_frames=32, seq_len=8, temporal_stride=4):
            Fake clip positions in its sorted 32-frame list: [0, 4, 8, …, 28]
            Real frames at those same positions are returned, regardless
            of what their actual filename integers are.
        """
        fake_total = self._fake_frame_counts.get(vid_key, 0)
        real_total = len(real_frames_sorted)

        if fake_total == real_total:
            # Same total count → map by position
            real_clip: list[dict] = []
            for k in range(self.seq_len):
                real_pos = start_pos + k * self.temp_stride
                if real_pos >= real_total:
                    return None  # Position out of bounds → drop
                real_clip.append(real_frames_sorted[real_pos])
            return real_clip
        else:
            # Different counts → fall back to strict_intersect
            logger.debug(
                f"index_aligned fallback to strict_intersect for vid_key={vid_key}: "
                f"fake_total={fake_total}, real_total={real_total}"
            )
            return self._align_strict_intersect(fake_clip, real_frames_sorted)
