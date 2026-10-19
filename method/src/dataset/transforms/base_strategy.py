"""
Abstract Augmentation Strategy Interface for DeepfakeWorkbench.

ARCHITECTURE PHILOSOPHY:
=======================
Augmentation Strategies define HOW to transform data. Instead of hardcoding
augmentation pipelines, we use pluggable strategies that can be swapped via config.

Strategies do NOT:
  - Load data (that's the Dataset's job)
  - Define model architectures (that's Backbone/Head/Detector's job)
  - Compute losses (that's the Loss's job)

Strategies DO:
  - Build augmentation pipelines (Albumentations, Kornia, etc.)
  - Handle different augmentation philosophies (AutoAugment, RandAugment, Geometric, etc.)
  - Support both CPU (Albumentations) and GPU (Kornia) augmentations
  - Provide consistent interfaces for spatial and temporal transforms

Strategy Types:
  - **StandardStrategy**: Classic augmentations (flip, crop, color jitter)
  - **AutoAugmentStrategy**: Learned augmentation policies
  - **GeometricStrategy**: Specialized geometric transformations (center vs boundary extraction)
  - **TemporalStrategy**: Video-aware augmentations with temporal consistency
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

import albumentations as A
from albumentations.core.composition import BaseCompose

from src.configs import AugmentationConfig


class BaseAugmentationStrategy(ABC):
    """
    Abstract base class for augmentation strategies.

    DESIGN CONTRACT:
    ================
    Augmentation strategies build transformation pipelines that can be
    applied to images/videos. They abstract away the choice of augmentation
    library (Albumentations, Kornia, custom) and provide a unified interface.

    Required Interface:
    -------------------
    1. build_transform(config, mode) -> Callable
       - Builds augmentation pipeline for 'train', 'val', or 'test' mode
       - Returns callable that transforms images: transform(image=img) -> dict

    2. supports_gpu() -> bool
       - Returns True if strategy supports GPU-side augmentation

    Optional Features:
    ------------------
    - Temporal consistency for video data
    - ReplayCompose for paired augmentation
    - Custom GPU kernels via Kornia

    Typical Usage in Dataset:
    --------------------------
    ```python
    strategy = build_augmentation_strategy(config.augmentation)
    train_transform = strategy.build_transform(config, mode='train')
    val_transform = strategy.build_transform(config, mode='val')

    # In __getitem__:
    image = load_image(path)
    augmented = train_transform(image=image)['image']
    ```

    Example Implementations:
    ------------------------
    ```python
    # Standard Albumentations Strategy
    class StandardAugmentationStrategy(BaseAugmentationStrategy):
        '''Classic CPU-based augmentations via Albumentations.'''

        def build_transform(self, config: AugmentationConfig, mode: str) -> Callable:
            if mode == 'train':
                return A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                    A.Normalize(mean=config.mean, std=config.std),
                    A.pytorch.ToTensorV2()
                ])
            else:  # val/test
                return A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.Normalize(mean=config.mean, std=config.std),
                    A.pytorch.ToTensorV2()
                ])

        def supports_gpu(self) -> bool:
            return False  # Albumentations runs on CPU

    # Geometric Strategy for OrthoPairGTL (using Kornia for GPU efficiency)
    class GeometricAugmentationStrategy(BaseAugmentationStrategy):
        '''
        Specialized strategy for extracting geometric regions.

        For OrthoPairGTL research: extracts face center vs outer boundary regions
        to detect artifacts at the blend boundary.

        Uses Kornia for GPU-accelerated, differentiable operations.
        '''

        def __init__(self):
            super().__init__()
            # Will use kornia.augmentation for GPU-side transforms

        def build_transform(self, config: AugmentationConfig, mode: str) -> Callable:
            import kornia as K
            import kornia.augmentation as KA

            # Build Kornia augmentation pipeline
            if mode == 'train':
                transform = nn.Sequential(
                    # Geometric region extraction
                    KA.CenterCrop(size=(config.center_size, config.center_size)),
                    # Standard augmentations
                    KA.RandomHorizontalFlip(p=0.5),
                    KA.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                    KA.Normalize(mean=config.mean, std=config.std)
                )
            else:
                transform = nn.Sequential(
                    KA.CenterCrop(size=(config.center_size, config.center_size)),
                    KA.Normalize(mean=config.mean, std=config.std)
                )

            return transform

        def extract_geometric_regions(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            '''
            Extract center crop and outer boundary regions.

            Args:
                x: Input tensor [B, C, H, W]

            Returns:
                Dict with 'center' and 'outer' tensors
            '''
            import kornia as K

            B, C, H, W = x.shape

            # Extract center region (face)
            center_size = H // 2  # Configurable
            center = K.geometry.transform.center_crop(
                x, (center_size, center_size)
            )

            # Extract outer boundary (4 squares around the center)
            # This is where deepfake blend artifacts often appear
            outer_regions = []

            # Top-left corner
            tl = x[:, :, :center_size//2, :center_size//2]
            # Top-right corner
            tr = x[:, :, :center_size//2, -center_size//2:]
            # Bottom-left corner
            bl = x[:, :, -center_size//2:, :center_size//2]
            # Bottom-right corner
            br = x[:, :, -center_size//2:, -center_size//2:]

            # Concatenate outer regions (could also average/max pool)
            # For now, concatenate along spatial dimension
            outer = torch.cat([tl, tr, bl, br], dim=0)  # [4*B, C, H', W']

            return {
                'center': center,
                'outer': outer
            }

        def supports_gpu(self) -> bool:
            return True  # Kornia runs on GPU

    # Temporal Strategy with Consistent Augmentation
    class TemporalAugmentationStrategy(BaseAugmentationStrategy):
        '''Video augmentation with temporal consistency via ReplayCompose.'''

        def build_transform(self, config: AugmentationConfig, mode: str) -> Callable:
            if mode == 'train':
                # Use ReplayCompose to apply same augmentation to all frames
                return A.ReplayCompose([
                    A.Resize(config.image_size, config.image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                    A.Normalize(mean=config.mean, std=config.std),
                    A.pytorch.ToTensorV2()
                ])
            else:
                return A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.Normalize(mean=config.mean, std=config.std),
                    A.pytorch.ToTensorV2()
                ])

        def supports_gpu(self) -> bool:
            return False
    ```

    Integration with OrthoPairGTL:
    -------------------------------
    For your geometric boundary artifact detection:

    ```python
    # In config.yaml:
    augmentation:
      strategy: geometric
      center_size: 112  # Half of 224
      extract_boundary: true
      num_outer_regions: 4

    # In detector:
    class GeometricDetector(BaseDetector):
        def __init__(self, config):
            super().__init__(config)
            self.geo_strategy = GeometricAugmentationStrategy()

        def forward(self, x):
            # Extract regions
            regions = self.geo_strategy.extract_geometric_regions(x)

            # Process each region through backbone
            center_feat = self.backbone(regions['center']).features
            outer_feat = self.backbone(regions['outer']).features

            # Project to embedding space
            center_embed = self.center_head(center_feat).output
            outer_embed = self.outer_head(outer_feat).output

            # OrthoPairGTL loss compares center vs outer embeddings
            return DetectorOutput(
                embeddings={'center': center_embed, 'outer': outer_embed},
                ...
            )
    ```
    """

    @abstractmethod
    def build_transform(self, config: AugmentationConfig, mode: str = "train") -> Callable[[Any], Any]:
        """
        Build augmentation pipeline for specified mode.

        Args:
            config: AugmentationConfig with transform settings
            mode: One of ['train', 'val', 'test']

        Returns:
            Callable transform that can be applied to images/videos
        """
        pass

    @abstractmethod
    def supports_gpu(self) -> bool:
        """
        Check if strategy supports GPU-side augmentation.

        Returns:
            True if transforms can run on GPU (e.g., via Kornia)
        """
        pass

    def build_replay_transform(self, config: AugmentationConfig, mode: str = "train") -> BaseCompose:
        """
        Build augmentation pipeline with replay support for temporal consistency.

        This is useful for video data where you want to apply the same
        random augmentation to all frames in a sequence.

        Args:
            config: AugmentationConfig with transform settings
            mode: One of ['train', 'val', 'test']

        Returns:
            ReplayCompose transform for consistent augmentation

        Raises:
            NotImplementedError: If strategy doesn't support replay
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support replay transforms")

    def extra_repr(self) -> str:
        """String representation for printing."""
        return f"gpu_support={self.supports_gpu()}"
