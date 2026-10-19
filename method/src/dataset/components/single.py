import logging
import random

import numpy as np
import torch

from src.dataset.core.base import DeepfakeBaseDataset

logger = logging.getLogger(__name__)


class StandardFrameDataset(DeepfakeBaseDataset):
    """
    Standard dataset for single-frame classification.

    When loaded from the Universal Manifest (JSONL), each item optionally
    includes masks, landmarks (106-pt InsightFace), and facemesh (478-pt
    MediaPipe) alongside the RGB frame.  Enable via `with_mask` /
    `with_landmark` constructor flags (inherited from base).

    Returns:
        {
            "image":     Tensor (C, H, W),
            "mask":      Tensor (H, W) uint8 | None,
            "landmarks": ndarray (106, 2) float32 | None,
            "facemesh":  ndarray (478, 3) float32 | None,
            "label":     int,
            "meta":      dict,
        }
    """

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = self._load_image(item["frame_path"])

        if self.transform:
            image = self.transform(image=image)["image"]

        result = {
            "image": image,
            "mask": None,
            "landmarks": None,
            "facemesh": None,
            "label": item["label"],
            "meta": {
                "video_id": item.get("jsonl_id") or item["video_id"],
                "task": item["task"],
                "path": item["frame_path"],
                "dataset_name": item.get("dataset_name", item["task"]),
            },
        }

        # Optionally load auxiliary modalities when JSONL media info is present
        if item.get("media"):
            if self.with_mask:
                mask = self._load_mask(item)
                if mask is not None:
                    result["mask"] = torch.from_numpy(mask)

            if self.with_landmark:
                result["landmarks"] = self._load_landmarks(item)
                result["facemesh"] = self._load_facemesh(item)

        return result


class StandardSequenceDataset(DeepfakeBaseDataset):
    """
    Sequence dataset for video clip classification.

    Builds temporally-strided clips via a sliding window.  Consistent
    augmentation is replayed across all frames in a clip.

    When loaded from the Universal Manifest, optionally yields per-frame
    masks, landmarks, and facemesh stacked along the temporal dimension.

    Args:
        sequence_length:          Number of frames per clip (T).
        temporal_stride:          Step size between frames within a clip.
        window_step:              Step size between clip start positions.
        max_sequences_per_video:  Cap on clips per video.

    Returns:
        {
            "image":     Tensor (T, C, H, W),
            "mask":      Tensor (T, H, W) | None,
            "landmarks": ndarray (T, 106, 2) float32 | None,
            "facemesh":  ndarray (T, 478, 3) float32 | None,
            "label":     int,
            "meta":      dict,
        }
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
        **kwargs,
    ):
        # Force scan_mode='all' so we get every frame to build sequences
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

        # 1. Group by (task, video_id) to prevent cross-task collisions.
        #    In FF++ the same video_id (e.g. "730_815") appears in all four
        #    forgery types — grouping by video_id alone silently discards ~75%
        #    of fake frames via dict key overwrites.
        self.video_map: dict[tuple, list] = {}
        for s in self.samples:
            vid_key = (s.get("task", "unknown"), s["video_id"])
            if vid_key not in self.video_map:
                self.video_map[vid_key] = []
            self.video_map[vid_key].append(s)

        # 2. Build clip index (strict temporal stride, sliding window)
        self.clips: list[list[dict]] = []
        for (_task, _vid), frames in self.video_map.items():
            frames.sort(key=lambda x: self._extract_frame_index(x["frame_name"]))
            frame_idx_map = {self._extract_frame_index(f["frame_name"]): f for f in frames}

            curr_clips: list[list[dict]] = []
            for i in range(0, len(frames), self.win_step):
                start_idx = self._extract_frame_index(frames[i]["frame_name"])
                clip: list[dict] = []
                valid = True
                for k in range(self.seq_len):
                    target_idx = start_idx + k * self.temp_stride
                    if target_idx in frame_idx_map:
                        clip.append(frame_idx_map[target_idx])
                    else:
                        valid = False
                        break
                if valid:
                    curr_clips.append(clip)

            if self.max_seq and len(curr_clips) > self.max_seq:
                curr_clips = random.sample(curr_clips, self.max_seq)
            self.clips.extend(curr_clips)

        logger.info(
            f"[{self.dataset_name}] StandardSequenceDataset: "
            f"{len(self.clips)} clips from {len(self.video_map)} videos"
        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_meta = self.clips[idx]

        # Edge padding: repeat last frame to fill short clips
        if len(clip_meta) < self.seq_len:
            pad = self.seq_len - len(clip_meta)
            clip_meta = clip_meta + [clip_meta[-1]] * pad

        first_frame = clip_meta[0]
        start_frame_idx = self._extract_frame_index(first_frame["frame_name"])
        label = first_frame["label"]
        task = first_frame["task"]

        # ── Load frames with consistent replay augmentation ───────────────────
        images = []
        replay_params = None

        first_img = self._load_image(clip_meta[0]["frame_path"])
        if self.transform:
            res = self.transform(image=first_img)
            images.append(res["image"])
            replay_params = res.get("replay")
            for m in clip_meta[1:]:
                img = self._load_image(m["frame_path"])
                if replay_params:
                    res = self.transform.replay(replay_params, image=img)
                else:
                    res = self.transform(image=img)
                images.append(res["image"])
        else:
            images.append(first_img)
            for m in clip_meta[1:]:
                images.append(self._load_image(m["frame_path"]))

        if isinstance(images[0], torch.Tensor):
            final_images = torch.stack(images)  # (T, C, H, W)
        else:
            final_images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)

        result = {
            "image": final_images,  # (T, C, H, W)
            "mask": None,
            "landmarks": None,
            "facemesh": None,
            "label": label,
            "meta": {
                "task": task,
                "video_id": first_frame.get("jsonl_id") or first_frame["video_id"],
                "dataset_name": first_frame.get("dataset_name", task),
                "sequence_start_frame": start_frame_idx,
                "sequence_length": len(clip_meta),
                "temporal_stride": self.temp_stride,
            },
        }

        # ── Optionally load auxiliary modalities ──────────────────────────────
        if first_frame.get("media"):
            if self.with_mask:
                masks = []
                for m in clip_meta:
                    mask = self._load_mask(m)
                    masks.append(torch.from_numpy(mask) if mask is not None else None)
                if all(mk is not None for mk in masks):
                    result["mask"] = torch.stack(masks)  # (T, H, W)

            if self.with_landmark:
                lms, meshes = [], []
                for m in clip_meta:
                    lms.append(self._load_landmarks(m))
                    meshes.append(self._load_facemesh(m))
                if all(lm is not None for lm in lms):
                    result["landmarks"] = np.stack(lms)   # (T, 106, 2)
                if all(fm is not None for fm in meshes):
                    result["facemesh"] = np.stack(meshes)  # (T, 478, 3)

        return result
