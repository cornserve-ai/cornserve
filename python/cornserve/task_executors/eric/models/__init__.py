from dataclasses import dataclass

from cornserve.task_executors.eric.schema import Modality


@dataclass
class ModelInfo:
    """Model info for a modality."""

    # Name of the model class
    class_name: str

    # Prefix of the model weights to collect
    weight_prefix: str


@dataclass
class RegistryEntry:
    """Registry entry for a model class."""

    # Name of module within `models`
    module: str

    # Modality to model info mapping
    model: dict[Modality, ModelInfo]


# Keyed by a model's type (usually its HF config `model_type`).
MODEL_REGISTRY: dict[str, RegistryEntry] = {
    "qwen2_vl": RegistryEntry(
        module="qwen2_vl",
        model={
            Modality.IMAGE: ModelInfo(
                class_name="Qwen2VisionTransformer",
                weight_prefix="visual.",
            ),
        },
    ),
}
