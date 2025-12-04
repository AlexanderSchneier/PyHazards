from .backbones import CNNPatchEncoder, MLPBackbone, TemporalEncoder
from .builder import build_model, default_builder
from .heads import ClassificationHead, RegressionHead, SegmentationHead
from .registry import available_models, register_model

__all__ = [
    "build_model",
    "available_models",
    "register_model",
    "MLPBackbone",
    "CNNPatchEncoder",
    "TemporalEncoder",
    "ClassificationHead",
    "RegressionHead",
    "SegmentationHead",
]

# Register default backbones
register_model("mlp", default_builder, defaults={"hidden_dim": 256, "depth": 2})
register_model("cnn", default_builder, defaults={"hidden_dim": 64, "in_channels": 3})
register_model("temporal", default_builder, defaults={"hidden_dim": 128, "num_layers": 1})
