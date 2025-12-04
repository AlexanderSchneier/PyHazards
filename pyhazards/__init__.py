from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("PyHazards")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback

from .datasets import (
    DataBundle,
    DataSplit,
    Dataset,
    FeatureSpec,
    LabelSpec,
    available_datasets,
    load_dataset,
    register_dataset,
)
from .models import (
    CNNPatchEncoder,
    ClassificationHead,
    MLPBackbone,
    RegressionHead,
    SegmentationHead,
    TemporalEncoder,
    available_models,
    build_model,
    register_model,
)
from .metrics import ClassificationMetrics, MetricBase, RegressionMetrics, SegmentationMetrics
from .engine import Trainer

__all__ = [
    "__version__",
    "DataBundle",
    "DataSplit",
    "Dataset",
    "FeatureSpec",
    "LabelSpec",
    "available_datasets",
    "load_dataset",
    "register_dataset",
    "CNNPatchEncoder",
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead",
    "MLPBackbone",
    "TemporalEncoder",
    "available_models",
    "build_model",
    "register_model",
    "Trainer",
    "MetricBase",
    "ClassificationMetrics",
    "RegressionMetrics",
    "SegmentationMetrics",
]
