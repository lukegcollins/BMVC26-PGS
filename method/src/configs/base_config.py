"""
Dataclass-based configuration objects for DeepfakeWorkbench.

All configuration is defined through dataclasses for:
- Type safety and IDE support
- Easy serialization to/from YAML
- Clear documentation of all parameters
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BackboneConfig:
    """Configuration for backbone network."""

    name: str = "dinov2_vitl14"
    weights_dir: str | None = None
    freeze_mode: str = "full"  # 'full', 'layernorm', 'none'
    l2_normalize: bool = True
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    force_fp32: bool = False
    lora: dict[str, Any] | None = None  # LoRA config: {rank, alpha, dropout, target_modules}
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadConfig:
    """Configuration for classification head."""

    type: str = "linear"  # 'linear', 'mlp', 'attention'
    hidden_dim: int = 2048
    dropout: float = 0.0
    num_classes: int = 2
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Configuration for loss functions."""

    weighting_mode: str = "static"  # 'static', 'dynamic', 'uncertainty'
    alpha: float = 1.0  # Cross-entropy weight
    beta: float = 0.5  # Orthogonality weight
    gamma: float = 0.0  # GTL weight
    use_gtl: bool = False
    use_axis_margin: bool = True
    use_residual_uniformity: bool = True
    use_paired_slerp_aug: bool = False
    ce_only_in_eval: bool = True
    debug_log_params: bool = False
    residual_loss: str = "l2"  # 'l1', 'l2', 'cosine'
    uniformity_t: float = 2.0
    axis_margin: float = 1.5
    residual_uniformity_t: float = 3.0
    gtl_disc_weight: float = 1.0
    gtl_small_weight: float = 0.2
    gtl_hidden_dim: int = 512
    gtl_output_dim: int = 256
    use_hard_rr_negatives: bool = True
    slerp_pair_multiplier: int = 1

    # KOT (Kinematic Orthogonality Training) parameters
    delta: float = 0.0  # Orthogonality loss weight (0 = disabled, backward compat)
    ortho_margin: float = 0.1  # Cosine similarity tolerance before penalty
    magnitude_tau: float = 0.05  # Min displacement magnitude to prevent collapse
    spatial_ratio: float = 0.5  # Ratio of spatial vs temporal pairs per batch

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationItem:
    """Single augmentation configuration."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentations."""

    training: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)
    testing: list[dict[str, Any]] | None = None  # Falls back to validation if None


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    type: str = "adam"  # 'adam', 'adamw', 'sgd'
    lr: float = 5e-5
    weight_decay: float = 1e-4
    # Adam-specific
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    # SGD-specific
    momentum: float = 0.9
    nesterov: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    type: str = "warmup_cosine_cyclic"
    warmup_epochs: int = 1
    cycle_epochs: int = 25
    cycles: int = 1
    lr_base: float = 5e-5
    lr_max: float = 5e-5
    lr_min: float = 1e-6
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving."""

    metric: str = "video_auc"
    aggregation: str = "macro"  # 'macro', 'micro', 'weighted'
    save_per_dataset: bool = True
    save_average: bool = True
    save_legacy: bool = False
    save_last: bool = True
    save_top_k: int = 3


@dataclass
class TemporalSamplingConfig:
    """Configuration for temporal/sequence sampling in video datasets."""

    # Sequence parameters
    sequence_length: int = 16  # Number of frames per clip (T)
    temporal_stride: int = 1  # Step size between frames within a clip (dilation)
    window_step: int = 8  # Step size between starting frames of clips (sliding window)
    max_sequences_per_video: int = 10  # Cap on clips per video to prevent imbalance

    # Pairing settings (for paired datasets)
    pairing_mode: str = "strict"  # 'strict' (align to source) or 'random_source'
    sync_augmentations: bool = True  # Apply same augmentations to paired frames
    sync_threshold: int = 5  # Frame search threshold for alignment

    # Video-level aggregation
    aggregation_method: str = "mean"  # 'mean', 'max', 'median', 'voting'
    aggregation_threshold: float = 0.5  # Threshold for voting aggregation


@dataclass
class DatasetSpec:
    """
    Specification for a single dataset in the Universal Manifest system.

    Describes which dataset to load, from which split, and with what filters.
    Used in train_datasets / val_datasets / test_datasets lists.

    Fields
    ------
    name            Dataset name (must match the manifest directory name).
    split           Split to load: train/training, val/validation, test/testing
                    (aliases are normalised in __post_init__).
    split_ratio     Manifest subdirectory, e.g. T-V-T, 80-20-T, 0-0-100.
                    When set, resolves to:
                      dataset_manifest_root/{name}/{processing_group}/{split_ratio}/
                    When omitted, the datamodule auto-detects a single subdir.
    log_name        Human-readable display name used for logging and report
                    columns. Allows multiple entries for the same dataset
                    (e.g. FF-ORIGINAL-C23, FF-DF-C23) to be tracked separately.
    methods         Label-key or method-name filter.  "all" (default) loads
                    everything.  An empty list [] is treated as "all".
                    Supports both old method names ("Deepfakes") and new
                    label_key values ("FF-FS_DF").
    compression     Optional compression filter (e.g. "c23", "c0").
    processing_group Override the global default_processing_group for this entry.

    Example YAML:
        test_datasets:
          - name: FaceForensics++
            log_name: FF-ORIGINAL-C23
            split: test
            split_ratio: T-V-T
            compression: c23
            methods: [FF-REAL, FF-FS_DF, FF-FR_F2F, FF-FS_FS, FF-FR_NT]
    """

    name: str = ""
    split: str = "train"
    split_ratio: str | None = None   # manifest subdirectory (T-V-T, 80-20-T, …)
    log_name: str | None = None      # display name for logging / report columns
    methods: list[str] | str = field(default_factory=lambda: "all")
    compression: str | None = None
    # Override the global default_processing_group for this specific dataset entry.
    processing_group: str | None = None
    # Set False to suppress label/path sanity-check warnings for datasets whose
    # label semantics are intentionally non-standard (e.g. FAVC-RVRA labeled 0
    # despite being in manipulated_content/, IDForge audio-only tasks labeled 0).
    integrity_checks: bool = True

    def __post_init__(self):
        # Normalise split aliases: training→train, validation→val, testing→test
        _aliases = {"training": "train", "validation": "val", "testing": "test"}
        self.split = _aliases.get(self.split.lower(), self.split)

        # Normalise methods: empty list [] → "all"; lone non-"all" string → list
        if isinstance(self.methods, list) and len(self.methods) == 0:
            self.methods = "all"
        elif isinstance(self.methods, str) and self.methods != "all":
            self.methods = [self.methods]

    @property
    def display_name(self) -> str:
        """log_name if set, otherwise falls back to name."""
        return self.log_name or self.name

    @property
    def methods_filter_active(self) -> bool:
        """True when a specific methods filter is set (not "all")."""
        return self.methods != "all"


def _to_dataset_specs(items: list) -> list[DatasetSpec]:
    """
    Normalise a list of dataset entries to DatasetSpec instances.

    Handles three input formats:
    - str: legacy name-only format → DatasetSpec(name=str)
    - dict: keyword args for DatasetSpec constructor (hyphenated keys are
            normalised to underscores so that e.g. split-ratio works in YAML)
    - DatasetSpec: already converted, pass through
    """
    result = []
    for item in items:
        if isinstance(item, DatasetSpec):
            result.append(item)
        elif isinstance(item, str):
            result.append(DatasetSpec(name=item))
        elif isinstance(item, dict):
            # Normalise hyphenated YAML keys → underscored Python kwargs
            normalised = {k.replace("-", "_"): v for k, v in item.items()}
            result.append(DatasetSpec(**normalised))
        else:
            raise TypeError(f"Cannot convert {type(item)} to DatasetSpec: {item}")
    return result


@dataclass
class DataConfig:
    """Configuration for datasets."""

    # ── Universal Manifest paths ─────────────────────────────────────────────
    # Root directory containing {dataset_name}/{processing_group}/
    # dataset_manifest.json + items.jsonl + stats.json
    dataset_manifest_root: str = "configs/datasets/manifests/rgb"
    # Default processing group (subfolder) used for all datasets unless
    # overridden per DatasetSpec.
    default_processing_group: str = "256w_256h_1.3x_32f_insightface"

    # ── Dataset type (per split) ─────────────────────────────────────────────
    # Training pairs real+fake via our contrastive loss → always "paired".
    # Validation/testing uses independent single-video classification → "single".
    #
    # Legacy: dataset_type applies globally.  New: use train_dataset_type,
    # val_dataset_type, test_dataset_type for per-split control.
    dataset_type: str = "standard"  # legacy global fallback
    train_dataset_type: str = "paired"  # 'paired', 'paired_sequence', 'sequence', 'single', 'dual_paired'
    val_dataset_type: str = "single"   # 'single', 'sequence'
    test_dataset_type: str = "single"  # 'single', 'sequence'

    # ── Temporal alignment for paired loading ────────────────────────────────
    # Controls how real-frame indices are matched to fake-frame indices when
    # re-encoding drift causes the same scene timestamp to fall on slightly
    # different frame numbers.
    #
    # strict_intersect : only use frames whose filename integer exists in BOTH
    #                    the real and fake directories.
    # tolerance        : match frames within ±temporal_alignment_tolerance
    #                    using a two-pointer approach (closest available).
    # index_aligned    : if |real_frames| == |fake_frames|, zip by sorted
    #                    position (0th↔0th, 1st↔1st …) ignoring filename
    #                    integers.  Else fallback to strict_intersect.
    temporal_alignment_mode: str = "index_aligned"
    temporal_alignment_tolerance: int = 2  # used only for 'tolerance' mode

    # ── Dataset lists ────────────────────────────────────────────────────────
    # Accept DatasetSpec objects, plain dicts, or legacy name strings.
    # __post_init__ normalises all entries to DatasetSpec.
    train_datasets: list[Any] = field(default_factory=list)
    val_datasets: list[Any] = field(default_factory=list)
    test_datasets: list[Any] = field(default_factory=list)

    # ── Legacy fields (kept for backward compatibility) ───────────────────────
    dataset_json_folder: str = "preprocessing/extended/dataset_json"
    compression: str = "c23"
    train_batch_size: int = 32
    test_batch_size: int = 32
    workers: int = 16
    frame_num_train: int = 32
    frame_num_val: int = 32
    frame_num_test: int = 32
    resolution: int = 224
    with_mask: bool = False
    with_landmark: bool = False
    use_data_augmentation: bool = True
    lmdb: bool = False
    scan_mode: str = "balanced"
    max_frames_per_video: int = 50

    # Label mapping for modality-aware class_id override.
    # When set, class_id in items.jsonl is overridden by this YAML at load time.
    # Required for datasets where visual vs. audio-visual labels differ
    # (e.g. IDForge-v1 audio-only tasks = 0 for visual models, 1 for AV models;
    #  FakeAVCeleb RVFA = 0 for visual, 1 for AV).
    # Path is relative to the working directory (project root).
    label_mapping_file: str | None = None   # e.g. "configs/datasets/labels/label_mappings_visual.yaml"
    label_type: str = "Binary"              # "Binary" or "MultiClass"

    # KOT: temporal (KiMoI) training datasets
    train_temporal_datasets: list[Any] = field(default_factory=list)

    # GPU-side normalization: when True, ImageNet normalization is skipped in
    # the CPU Albumentations pipeline and applied on GPU via
    # on_after_batch_transfer.  Frees CPU cycles for I/O and spatial augments.
    gpu_normalize: bool = False

    # Video-grouped batch sampler (required for GSD set-geometric losses).
    # Groups frames by fake video ID so L_concentration can compute per-video
    # C_dir statistics.  Effective batch size = num_videos × frames_per_video.
    use_video_grouped_sampler: bool = False
    gsd_num_videos_per_batch: int = 8   # distinct fake videos per batch
    gsd_frames_per_video: int = 4       # frames sampled per video per batch

    # Temporal/sequence sampling configuration
    temporal: TemporalSamplingConfig = field(default_factory=TemporalSamplingConfig)

    def __post_init__(self):
        """Convert nested dicts to dataclass instances if needed."""
        if isinstance(self.temporal, dict):
            self.temporal = TemporalSamplingConfig(**self.temporal)

        # Normalise dataset list entries to DatasetSpec
        self.train_datasets = _to_dataset_specs(self.train_datasets)
        self.val_datasets = _to_dataset_specs(self.val_datasets)
        self.test_datasets = _to_dataset_specs(self.test_datasets)
        self.train_temporal_datasets = _to_dataset_specs(self.train_temporal_datasets)


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    start_epoch: int = 0
    num_epochs: int = 25
    seed: int = 42
    grad_accum_steps: int = 4
    grad_clip: float = 2.0
    nan_check_frequency: int = 500
    max_nan_tolerance: int = 5
    use_amp: bool = True
    use_compile: bool = True
    compile_mode: str = "default"  # 'default', 'reduce-overhead', 'max-autotune'
    cudnn_benchmark: bool = True
    deterministic: bool = False
    ddp: bool = False
    local_rank: int = 0


@dataclass
class AutoTestConfig:
    """
    Configuration for automatic testing after training.

    checkpoint options
    ------------------
    'best'       Macro-val-AUC best checkpoint (epoch=*-video_auc=*.ckpt).
    'last'       last.ckpt saved at the final training step.
    'best_all'   All per-dataset best_*.ckpt files PLUS the macro-best.
                 Runs trainer.test() once per checkpoint and writes a separate
                 test/<ckpt_stem>/ results directory for each.
    <path>       Explicit path to any .ckpt file.
    """

    enabled: bool = False
    checkpoint: str = "best"  # 'best', 'last', 'best_all', or explicit path
    test_after_epoch: int | None = None
    # Override frame_num_test for inline auto_test runs.
    # The full benchmark uses evaluate_checkpoints.py with frame_num=32.
    # Set to e.g. 8 here for a quick post-training sanity check (4× faster).
    # None = use DataConfig.frame_num_test unchanged.
    frame_num: int | None = None


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    This is the top-level configuration object that aggregates all sub-configs
    and corresponds to the full experiment YAML structure.
    """

    # Experiment identification
    model_name: str = "orthoflow"
    task_target: str = "experiment"
    log_dir: str = "./training/logs"

    # Component configurations
    backbone_name: str = "dinov2"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    auto_test: AutoTestConfig = field(default_factory=AutoTestConfig)

    # Loss routing (v2 fix: allows bypassing CombinedLoss)
    # "combined" = CombinedLoss (legacy), "orthoflow_gtl" = OrthoPairGTLLoss direct, "ce" = CE only
    loss_type: str = "combined"

    # Feature flags
    save_ckpt: bool = True
    save_feat: bool = True
    metric_scoring: str = "video_auc"

    def __post_init__(self):
        """Convert nested dicts to dataclass instances if needed."""
        if isinstance(self.backbone, dict):
            self.backbone = BackboneConfig(**self.backbone)
        if isinstance(self.head, dict):
            self.head = HeadConfig(**self.head)
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)
        if isinstance(self.checkpoint, dict):
            self.checkpoint = CheckpointConfig(**self.checkpoint)
        if isinstance(self.augmentations, dict):
            self.augmentations = AugmentationConfig(**self.augmentations)
        if isinstance(self.auto_test, dict):
            self.auto_test = AutoTestConfig(**self.auto_test)


def _flatten_config(config: dict, flat_mapping: dict) -> dict:
    """
    Convert flat YAML config to nested structure using mapping.

    Handles the user's preferred flat config format (backbone_name, backbone_config, etc.)
    and converts to the nested structure expected by ExperimentConfig.
    """
    result = {}

    # Direct mappings
    direct_keys = ["model_name", "task_target", "log_dir", "save_ckpt", "save_feat", "metric_scoring", "loss_type"]
    for key in direct_keys:
        if key in config:
            result[key] = config[key]

    # Backbone
    if "backbone_name" in config:
        result["backbone_name"] = config["backbone_name"]
    if "backbone_config" in config:
        result["backbone"] = config["backbone_config"]

    # Head
    if "head_config" in config:
        result["head"] = config["head_config"]

    # Loss
    if "loss_config" in config:
        result["loss"] = config["loss_config"]

    # Augmentations
    if "augmentations" in config:
        result["augmentations"] = config["augmentations"]

    # Checkpoint
    if "checkpoint_config" in config:
        result["checkpoint"] = config["checkpoint_config"]

    # Auto test
    if "auto_test" in config:
        result["auto_test"] = config["auto_test"]

    # Data config - map from flat keys
    data_config = {}

    # ── Universal Manifest fields (new) ──────────────────────────────────────
    universal_data_keys = [
        "dataset_manifest_root",
        "default_processing_group",
        "train_dataset_type",
        "val_dataset_type",
        "test_dataset_type",
        "temporal_alignment_mode",
        "temporal_alignment_tolerance",
        "label_mapping_file",
        "label_type",
    ]
    for key in universal_data_keys:
        if key in config:
            data_config[key] = config[key]

    # New-style dataset lists (list of dicts/DatasetSpec) passed directly
    for list_key in ("train_datasets", "val_datasets", "test_datasets"):
        if list_key in config:
            data_config[list_key] = config[list_key]

    # ── Legacy flat data keys ─────────────────────────────────────────────────
    data_mapping = {
        "dataset_type": "dataset_type",
        "dataset_json_folder": "dataset_json_folder",
        "train_dataset": "train_datasets",   # legacy list[str] format
        "val_dataset": "val_datasets",
        "test_dataset": "test_datasets",
        "compression": "compression",
        "train_batchSize": "train_batch_size",
        "test_batchSize": "test_batch_size",
        "workers": "workers",
        "resolution": "resolution",
        "with_mask": "with_mask",
        "with_landmark": "with_landmark",
        "use_data_augmentation": "use_data_augmentation",
        "lmdb": "lmdb",
        "gpu_normalize": "gpu_normalize",
        "use_video_grouped_sampler": "use_video_grouped_sampler",
        "gsd_num_videos_per_batch": "gsd_num_videos_per_batch",
        "gsd_frames_per_video": "gsd_frames_per_video",
    }
    for yaml_key, config_key in data_mapping.items():
        if yaml_key in config:
            data_config[config_key] = config[yaml_key]

    # KOT: temporal training datasets
    if "train_temporal_dataset" in config:
        data_config["train_temporal_datasets"] = config["train_temporal_dataset"]

    # Frame num handling
    if "frame_num" in config:
        frame_num = config["frame_num"]
        if isinstance(frame_num, dict):
            data_config["frame_num_train"] = frame_num.get("train", 32)
            data_config["frame_num_val"] = frame_num.get("val", 32)
            data_config["frame_num_test"] = frame_num.get("test", 32)
        else:
            data_config["frame_num_train"] = frame_num
            data_config["frame_num_val"] = frame_num
            data_config["frame_num_test"] = frame_num

    # Temporal sampling config - map from flat keys
    temporal_config = {}
    temporal_mapping = {
        "sequence_length": "sequence_length",
        "temporal_stride": "temporal_stride",
        "window_step": "window_step",
        "max_sequences_per_video": "max_sequences_per_video",
        "pairing_mode": "pairing_mode",
        "sync_augmentations": "sync_augmentations",
        "sync_threshold": "sync_threshold",
        "aggregation_method": "aggregation_method",
        "aggregation_threshold": "aggregation_threshold",
    }
    for yaml_key, config_key in temporal_mapping.items():
        if yaml_key in config:
            temporal_config[config_key] = config[yaml_key]

    # Also check for nested temporal_config section
    if "temporal_config" in config:
        temporal_config.update(config["temporal_config"])

    if temporal_config:
        data_config["temporal"] = temporal_config

    if data_config:
        result["data"] = data_config

    # Training config - map from flat keys
    training_config = {}
    training_mapping = {
        "start_epoch": "start_epoch",
        "nEpochs": "num_epochs",
        "manualSeed": "seed",
        "grad_accum_steps": "grad_accum_steps",
        "grad_clip": "grad_clip",
        "nan_check_frequency": "nan_check_frequency",
        "max_nan_tolerance": "max_nan_tolerance",
        "use_amp": "use_amp",
        "use_compile": "use_compile",
        "compile_mode": "compile_mode",
        "cudnn": "cudnn_benchmark",
        "ddp": "ddp",
        "local_rank": "local_rank",
    }
    for yaml_key, config_key in training_mapping.items():
        if yaml_key in config:
            training_config[config_key] = config[yaml_key]

    if training_config:
        result["training"] = training_config

    # Optimizer config
    if "optimizer" in config:
        opt_raw = config["optimizer"]
        opt_type = opt_raw.get("type", "adam")
        opt_config = {"type": opt_type}

        # Merge type-specific params
        if opt_type in opt_raw:
            opt_config.update(opt_raw[opt_type])

        # Preserve optimizer-level keys that aren't the type or type-specific dict
        # (e.g. backbone_lr_multiplier, head_lr, etc.)
        extra = opt_config.get("extra", {})
        for k, v in opt_raw.items():
            if k not in ("type", opt_type) and k not in opt_config:
                if k == "extra" and isinstance(v, dict):
                    # Merge nested extra dict contents rather than double-nesting
                    extra.update(v)
                else:
                    extra[k] = v
        if extra:
            opt_config["extra"] = extra

        result["optimizer"] = opt_config

    # Scheduler config - map from flat keys
    scheduler_config = {}
    scheduler_mapping = {
        "lr_scheduler": "type",
        "lr_warmup_epochs": "warmup_epochs",
        "lr_cycle_epochs": "cycle_epochs",
        "lr_cycles": "cycles",
        "lr_base": "lr_base",
        "lr_max": "lr_max",
        "lr_min": "lr_min",
    }
    for yaml_key, config_key in scheduler_mapping.items():
        if yaml_key in config:
            scheduler_config[config_key] = config[yaml_key]

    if scheduler_config:
        result["scheduler"] = scheduler_config

    return result


def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.

    Supports both the user's preferred flat format and nested format.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ExperimentConfig instance

    Example:
        config = load_config("configs/experiments/my_experiment.yaml")
        print(config.backbone.name)
    """
    config_path = Path(config_path)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Detect format: if top-level has 'backbone_config' or 'loss_config', it's flat format.
    # New universal manifest keys (train_dataset_type etc.) are also treated as flat format.
    flat_keys = {
        "backbone_config", "head_config", "loss_config", "train_dataset", "val_dataset",
        # New universal manifest top-level keys
        "dataset_manifest_root", "train_dataset_type", "val_dataset_type",
        "temporal_alignment_mode",
    }
    is_flat = bool(flat_keys & set(raw_config.keys()))

    if is_flat:
        # Convert flat format to nested
        nested_config = _flatten_config(raw_config, {})
    else:
        nested_config = raw_config

    return ExperimentConfig(**nested_config)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """
    Save experiment configuration to YAML file.

    Args:
        config: ExperimentConfig instance
        path: Output path for YAML file
    """
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
