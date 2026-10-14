"""Small, controlled reference folding architectures and training pipeline."""

from .models import ReferenceFoldingConfig, ReferenceFoldingModel, build_model

__all__ = ["ReferenceFoldingConfig", "ReferenceFoldingModel", "build_model"]
