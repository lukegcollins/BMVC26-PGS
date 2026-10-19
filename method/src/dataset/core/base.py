import json
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DeepfakeBaseDataset(Dataset, ABC):
    """
    Abstract Base Class for Deepfake Datasets.

    Supports two manifest formats:

    1. **Universal Manifest (JSONL)** — new format.
       Pass `manifest_dir` pointing to the directory that contains
       `dataset_manifest.json` and `items.jsonl`.  Optionally filter by
       `methods`, `compression`.

    2. **Legacy JSON** — old format.
       Pass `manifest_path` pointing to a single `.json` file with
       `{split: [video_entry, ...]}` structure.

    Features:
    - Streaming JSONL parser with split / method / compression filters
    - Directory scanning into per-frame samples
    - Rich sample records (include full JSONL media dict for modality loading)
    - Metadata caching (frame counts for ratio alignment)
    - Integrity checks (WARN only)
    - Clean init summary (video + frame label distributions)
    """

    def __init__(
        self,
        # ── New Universal Manifest params ────────────────────────────────────
        manifest_dir: str | None = None,
        methods: list[str] | str = "all",
        compression: str | None = None,
        # ── Legacy JSON param (backward compat) ──────────────────────────────
        manifest_path: str | None = None,
        # ── Common params ────────────────────────────────────────────────────
        split: str = "train",
        transform=None,
        scan_mode: str = "balanced",
        max_frames_per_video: int = 50,
        extensions: tuple = (".png", ".jpg", ".jpeg"),
        integrity_checks: bool = True,
        max_warnings: int = 10,
        with_mask: bool = False,
        with_landmark: bool = False,
        # ── Label mapping (modality-aware class_id override) ─────────────────
        # Flat dict: {label_key: class_id}  e.g. {"IDFv1-RVC_TM": 0, ...}
        # When provided, overrides class_id from items.jsonl for every item
        # whose label_key is present in the mapping.
        label_mapping: dict[str, int] | None = None,
    ):
        # Resolve a human-readable dataset name for logging
        if manifest_dir:
            _mdir = Path(manifest_dir)
            # e.g. .../FaceForensics++/256w_256h_1.3x_32f_insightface
            self.dataset_name = _mdir.parent.name
            self.manifest_path = str(_mdir / "items.jsonl")
        elif manifest_path:
            self.dataset_name = Path(manifest_path).stem
            self.manifest_path = str(manifest_path)
        else:
            raise ValueError("Provide either manifest_dir or manifest_path.")

        self.split = split
        self.transform = transform
        self.scan_mode = scan_mode
        self.max_frames = int(max_frames_per_video)
        self.extensions = tuple(e.lower() for e in extensions)
        self.integrity_checks = bool(integrity_checks)
        self.max_warnings = int(max_warnings)
        self.with_mask = with_mask
        self.with_landmark = with_landmark
        self.label_mapping = label_mapping

        self._meta_cache: dict[str, int] = {}
        # Dataset root resolved from dataset_manifest.json (JSONL path only)
        self._dataset_root: str | None = None

        # ── Load video entries ────────────────────────────────────────────────
        if manifest_dir:
            self.video_entries, self._dataset_root = self._load_from_jsonl(
                manifest_dir=manifest_dir,
                split=split,
                methods=methods,
                compression=compression,
            )
        else:
            self.video_entries = self._load_from_legacy_json(
                manifest_path=manifest_path,
                split=split,
            )

        if not self.video_entries:
            logger.warning(
                f"[{self.dataset_name}] No entries found for split='{split}' "
                f"(manifest={self.manifest_path})"
            )

        self._log_manifest_summary()

        # ── Build per-frame sample index ──────────────────────────────────────
        self.samples, scan_stats = self._scan_dataset_with_stats()
        logger.info(
            f"[{self.dataset_name}][{split.upper()}] Indexed {len(self.samples)} samples "
            f"from {scan_stats['videos_found']} videos "
            f"(manifest videos={len(self.video_entries)})."
        )
        if scan_stats["videos_missing"] > 0:
            logger.warning(
                f"[{self.dataset_name}][{split.upper()}] Missing video dirs: "
                f"{scan_stats['videos_missing']} (skipped)."
            )
        if scan_stats["videos_empty"] > 0:
            logger.warning(
                f"[{self.dataset_name}][{split.upper()}] Empty video dirs "
                f"(no frames): {scan_stats['videos_empty']} (skipped)."
            )

        self._log_frame_summary()

        if self.integrity_checks:
            self._run_integrity_checks(scan_stats)

    # =========================================================================
    # Manifest loading
    # =========================================================================

    def _load_from_jsonl(
        self,
        manifest_dir: str,
        split: str,
        methods: list[str] | str = "all",
        compression: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Stream items.jsonl and return (video_entries, dataset_root).

        Filtering rules
        ---------------
        - split       : only items where item["split"] == split
        - methods     : "all" → no filter; list → real (method="original") is
                        always included; fake is included only if its method is
                        in the list
        - compression : None → no filter; otherwise item compression must match
                        (items with compression=null/None are allowed through
                        regardless so DSv2 isn't accidentally excluded when a
                        compression filter is set for FF++; set compression on
                        the DatasetSpec for datasets that need it)
        """
        mdir = Path(manifest_dir)

        # ── Read dataset root from dataset_manifest.json ──────────────────────
        manifest_json_path = mdir / "dataset_manifest.json"
        if not manifest_json_path.exists():
            raise FileNotFoundError(
                f"dataset_manifest.json not found in {mdir}. "
                "Ensure the directory contains dataset_manifest.json and items.jsonl."
            )
        with open(manifest_json_path) as f:
            manifest_meta = json.load(f)
        dataset_root = Path(manifest_meta["dataset"]["root"])

        # ── Stream and filter items.jsonl ──────────────────────────────────────
        items_path = mdir / "items.jsonl"
        if not items_path.exists():
            raise FileNotFoundError(f"items.jsonl not found in {mdir}.")

        methods_set = None if methods == "all" else set(methods)
        entries: list[dict[str, Any]] = []

        with open(items_path) as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                item = json.loads(raw_line)

                # ── Filter: split ─────────────────────────────────────────────
                if item.get("split") != split:
                    continue

                attrs = item.get("attributes", {})
                item_method = attrs.get("method", "")
                item_compression = attrs.get("compression")  # may be None

                # ── Filter: methods ───────────────────────────────────────────
                # Real videos (method="original") always pass through so the
                # paired loader can find them regardless of which fake methods
                # are requested.
                # Fake videos are matched against EITHER the raw method name
                # (legacy: "Deepfakes") OR the label_key (new: "FF-FS_DF"),
                # so both old and new config formats work transparently.
                if methods_set is not None and item_method != "original":
                    item_label_key = attrs.get("label_key", "")
                    if item_method not in methods_set and item_label_key not in methods_set:
                        continue

                # ── Filter: compression ───────────────────────────────────────
                # Skip items with an explicit compression value that doesn't
                # match the requested one.  Items with compression=None
                # (e.g. DSv2) are never filtered by compression.
                if compression is not None and item_compression is not None:
                    if item_compression != compression:
                        continue

                # ── Build normalised video_entry ──────────────────────────────
                rgb_media = item.get("media", {}).get("rgb", {})
                frame_dir_rel = rgb_media.get("frame_dir", "")
                abs_frame_dir = str(dataset_root / frame_dir_rel) if frame_dir_rel else ""

                label = item["label"]["class_id"]
                # Override class_id from label mapping (modality-aware labels).
                # The mapping is the source of truth; the manifest value is a fallback.
                if self.label_mapping is not None:
                    lk = attrs.get("label_key", "")
                    if lk in self.label_mapping:
                        label = self.label_mapping[lk]
                    elif lk:
                        logger.warning(
                            f"[{Path(manifest_dir).parent.name}] label_key '{lk}' not in "
                            f"label_mapping — using manifest class_id={label}"
                        )

                entry = {
                    # ── Core fields (used everywhere) ─────────────────────────
                    "video_id": item["key"],
                    "label": label,
                    "task": attrs.get("task_abbr", attrs.get("method", "unknown")),
                    "path": abs_frame_dir,
                    "pair_id": _resolve_pair_id(item),
                    # ── Rich JSONL payload (used by single/paired loaders) ─────
                    "media": item.get("media"),
                    "attributes": attrs,
                    "dataset_root": str(dataset_root),
                    "jsonl_id": item.get("id"),
                }
                entries.append(entry)

        return entries, str(dataset_root)

    def _load_from_legacy_json(
        self,
        manifest_path: str,
        split: str,
    ) -> list[dict[str, Any]]:
        """
        Load video entries from the legacy per-split JSON format.

        Expected structure:
            {"train": [...], "val": [...], "test": [...]}
        or the older per-split flat list format.
        """
        full_manifest: dict = {}
        try:
            with open(manifest_path) as f:
                full_manifest = json.load(f)
        except Exception as e:
            logger.warning(
                f"[{self.dataset_name}] Failed to load manifest: "
                f"{manifest_path} ({e})"
            )
            return []

        entries = full_manifest.get(split, []) if isinstance(full_manifest, dict) else []
        if not entries:
            logger.warning(
                f"[{self.dataset_name}] No entries for split='{split}' in "
                f"{manifest_path}"
            )
        return entries

    # =========================================================================
    # Scanning
    # =========================================================================

    def _scan_dataset_with_stats(self) -> tuple[list[dict], dict[str, int]]:
        """Scan frame directories to build the playable per-frame sample list."""
        samples: list[dict] = []
        stats = {"videos_missing": 0, "videos_empty": 0, "videos_found": 0}

        for video_entry in self.video_entries:
            if "path" not in video_entry or "video_id" not in video_entry or "label" not in video_entry:
                logger.warning(
                    f"[{self.dataset_name}][{self.split.upper()}] Malformed video entry "
                    f"(missing keys): {video_entry}"
                )
                continue

            video_path = Path(video_entry["path"])
            if not video_path.exists():
                stats["videos_missing"] += 1
                continue

            stats["videos_found"] += 1

            frames = sorted(
                f.name
                for f in video_path.iterdir()
                if f.is_file() and f.suffix.lower() in self.extensions
            )
            if not frames:
                stats["videos_empty"] += 1
                continue

            if self.scan_mode == "balanced" and len(frames) > self.max_frames:
                indices = torch.linspace(0, len(frames) - 1, self.max_frames).long().tolist()
                selected_frames = [frames[i] for i in indices]
            elif self.scan_mode == "first_k":
                selected_frames = frames[: self.max_frames]
            else:
                selected_frames = frames

            for frame_name in selected_frames:
                samples.append(
                    {
                        "video_id": video_entry["video_id"],
                        "label": int(video_entry["label"]),
                        "task": video_entry.get("task", "unknown"),
                        "frame_path": str(video_path / frame_name),
                        "video_path": str(video_path),
                        "pair_id": video_entry.get("pair_id"),
                        "frame_name": frame_name,
                        # Rich JSONL payload — None for legacy JSON manifests
                        "media": video_entry.get("media"),
                        "attributes": video_entry.get("attributes"),
                        "dataset_root": video_entry.get("dataset_root"),
                        "jsonl_id": video_entry.get("jsonl_id"),
                    }
                )

        return samples, stats

    # =========================================================================
    # Summaries
    # =========================================================================

    def _log_manifest_summary(self) -> None:
        if not self.video_entries:
            return
        vid_labels = Counter(int(v.get("label", -1)) for v in self.video_entries)
        tasks = Counter(str(v.get("task", "unknown")) for v in self.video_entries)
        fake_total = sum(1 for v in self.video_entries if int(v.get("label", -1)) == 1)
        fake_missing_pair = sum(
            1 for v in self.video_entries
            if int(v.get("label", -1)) == 1 and not v.get("pair_id")
        )
        logger.info(
            f"[{self.dataset_name}] Manifest split='{self.split}' | "
            f"videos={len(self.video_entries)} | labels(video-level)={dict(vid_labels)} | "
            f"tasks(top)={tasks.most_common(3)} | "
            f"scan_mode={self.scan_mode} max_frames_per_video={self.max_frames}"
        )
        if fake_total > 0 and fake_missing_pair > 0:
            logger.warning(
                f"[{self.dataset_name}][{self.split.upper()}] "
                f"{fake_missing_pair}/{fake_total} fake videos missing pair_id "
                f"(may break strict pairing)."
            )

    def _log_frame_summary(self) -> None:
        if not self.samples:
            return
        frame_labels = Counter(int(s.get("label", -1)) for s in self.samples)
        n = len(self.samples)
        r0 = frame_labels.get(0, 0)
        r1 = frame_labels.get(1, 0)
        logger.info(
            f"[{self.dataset_name}][{self.split.upper()}] Frame samples={n} | "
            f"labels(frame-level)={{0:{r0} ({(r0/max(n,1))*100:.1f}%), "
            f"1:{r1} ({(r1/max(n,1))*100:.1f}%)}}"
        )

    # =========================================================================
    # Integrity checks (WARN ONLY)
    # =========================================================================

    def _run_integrity_checks(self, scan_stats: dict[str, int]) -> None:
        # 1) Label consistency per video_id
        labels_by_vid = defaultdict(set)
        for s in self.samples:
            labels_by_vid[str(s.get("video_id"))].add(int(s.get("label", -1)))
        inconsistent = [(vid, labs) for vid, labs in labels_by_vid.items() if len(labs) > 1]
        if inconsistent:
            logger.warning(
                f"[{self.dataset_name}][{self.split.upper()}] "
                f"{len(inconsistent)} videos with mixed labels."
            )
            for vid, labs in inconsistent[: self.max_warnings]:
                logger.warning(
                    f"[{self.dataset_name}][{self.split.upper()}] "
                    f"Mixed labels: video_id={vid} labels={sorted(labs)}"
                )

        # 2) Path semantic check (FF++ style tokens)
        tokens_present = 0
        suspicious = []
        for s in self.samples:
            combined = (str(s.get("video_path", "")) + " " + str(s.get("frame_path", ""))).lower()
            if "original_content" in combined or "manipulated_content" in combined:
                tokens_present += 1
                lab = int(s.get("label", -1))
                if lab == 0 and "original_content" not in combined:
                    suspicious.append((s.get("video_id"), lab, s.get("video_path")))
                if lab == 1 and "manipulated_content" not in combined:
                    suspicious.append((s.get("video_id"), lab, s.get("video_path")))
        if tokens_present > 0 and suspicious:
            logger.warning(
                f"[{self.dataset_name}][{self.split.upper()}] "
                f"Label/path semantics suspicious for {len(suspicious)} samples."
            )
            for vid, lab, vp in suspicious[: self.max_warnings]:
                logger.warning(
                    f"[{self.dataset_name}][{self.split.upper()}] "
                    f"Suspicious: video_id={vid} label={lab} path={vp}"
                )

        # 3) Single-class warning
        frame_labels = Counter(int(s.get("label", -1)) for s in self.samples)
        present = [k for k, v in frame_labels.items() if v > 0 and k in (0, 1)]
        if len(present) <= 1 and self.samples:
            logger.warning(
                f"[{self.dataset_name}][{self.split.upper()}] Only one class present: "
                f"{dict(frame_labels)}. AUC may be undefined."
            )

    # =========================================================================
    # Modality loading helpers
    # =========================================================================

    def _load_mask(self, sample: dict) -> np.ndarray | None:
        """
        Load binary mask PNG for a specific frame.

        Returns a (H, W) uint8 array (greyscale) or None if unavailable.
        """
        media = sample.get("media")
        dataset_root = sample.get("dataset_root")
        if not media or not dataset_root:
            return None
        mask_media = media.get("mask", {})
        mask_dir = mask_media.get("frame_dir")
        if not mask_dir:
            return None
        mask_path = Path(dataset_root) / mask_dir / sample["frame_name"]
        if not mask_path.exists():
            return None
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return img  # (H, W) uint8

    def _load_landmarks(self, sample: dict) -> np.ndarray | None:
        """
        Load 106-pt InsightFace landmarks for a specific frame from the NPZ.

        Returns (106, 2) float32 in 256×256 crop pixel space, or None.
        """
        media = sample.get("media")
        dataset_root = sample.get("dataset_root")
        if not media or not dataset_root:
            return None
        lm_media = media.get("landmarks", {})
        lm_path_rel = lm_media.get("path")
        if not lm_path_rel:
            return None
        lm_path = Path(dataset_root) / lm_path_rel
        if not lm_path.exists():
            return None
        try:
            data = np.load(str(lm_path), allow_pickle=False)
            key = sample["frame_name"]
            return data[key] if key in data else None
        except Exception as e:
            logger.debug(f"Failed to load landmarks {lm_path}: {e}")
            return None

    def _load_facemesh(self, sample: dict) -> np.ndarray | None:
        """
        Load 478-pt MediaPipe facemesh for a specific frame from the NPZ.

        Returns (478, 3) float32 in 256×256 crop pixel space, or None.
        """
        media = sample.get("media")
        dataset_root = sample.get("dataset_root")
        if not media or not dataset_root:
            return None
        fm_media = media.get("facemesh", {})
        fm_path_rel = fm_media.get("path")
        if not fm_path_rel:
            return None
        fm_path = Path(dataset_root) / fm_path_rel
        if not fm_path.exists():
            return None
        try:
            data = np.load(str(fm_path), allow_pickle=False)
            key = sample["frame_name"]
            return data[key] if key in data else None
        except Exception as e:
            logger.debug(f"Failed to load facemesh {fm_path}: {e}")
            return None

    # =========================================================================
    # Metadata + utilities
    # =========================================================================

    def _get_original_frame_count(self, video_path: str, video_id: str) -> int:
        """
        Retrieves original frame count from metadata.json for Ratio Alignment.
        Handles both New (Nested) and Legacy (Flat) schemas.
        """
        if video_id in self._meta_cache:
            return self._meta_cache[video_id]

        meta_path = Path(video_path) / "metadata.json"
        count = 0

        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                if (
                    isinstance(data, dict)
                    and "media" in data
                    and isinstance(data["media"], dict)
                    and "total_frames" in data["media"]
                ):
                    count = int(data["media"]["total_frames"])
                elif isinstance(data, dict) and "original_frames" in data:
                    count = int(data["original_frames"])
            except Exception as e:
                logger.warning(
                    f"[{self.dataset_name}] Failed to read metadata.json "
                    f"for video_id={video_id}: {e}"
                )

        self._meta_cache[video_id] = count
        return count

    def _extract_frame_index(self, frame_name: str) -> int:
        name = Path(frame_name).stem
        digits = re.findall(r"\d+", name)
        return int(digits[-1]) if digits else 0

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx):
        pass


# =============================================================================
# Module-level helpers
# =============================================================================

def _resolve_pair_id(item: dict) -> str | None:
    """
    Derive the paired real video's ID from a JSONL item's attributes.

    This is intentionally dataset-agnostic and follows a priority order:

    1. FF++  — `target_id` directly names the real video's key.
    2. DSv2  — composite `{identity_target}-{recording_target}` matches
               the real video key pattern `{identity}-{recording}`.
    3. DSv2 fallback — identity-only match if recording_target is absent.
    4. Real videos  — always return None (they have no pair).
    """
    if item["label"]["class_id"] == 0:
        return None  # Real videos are never the "requester" of a pair

    attrs = item.get("attributes", {})

    # FF++: target_id is the background real video's numeric key
    if "target_id" in attrs:
        return attrs["target_id"]

    # DSv2: composite identity-recording key
    if "identity_target" in attrs and "recording_target" in attrs:
        return f"{attrs['identity_target']}-{attrs['recording_target']}"

    # DSv2 fallback: identity only (matches any real video of that identity)
    if "identity_target" in attrs:
        return attrs["identity_target"]

    return None
