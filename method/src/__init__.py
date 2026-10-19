"""
DeepfakeWorkbench - A modular framework for deepfake detection research.

This package provides:
- Modular backbone architectures (DINOv2, DINOv3, timm models)
- Pluggable detection heads (Linear, MLP, Projection, Temporal)
- Multi-task loss functions (CE, Contrastive, OrthoPairGTL)
- Video-level metric aggregation
- PyTorch Lightning training pipeline
- Multi-dataset support with pluggable augmentation strategies

Quick Start:
    from src.configs import load_config
    from src.lightning import DeepfakeDetectorModule, DeepfakeDataModule

    config = load_config("configs/experiments/orthoflow.yaml")
    module = DeepfakeDetectorModule(config)
    data = DeepfakeDataModule(config)
"""

__version__ = "0.1.0"
