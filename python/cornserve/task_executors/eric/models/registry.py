"""All model-specific definitions.

All models are expected to be in the `models` subdirectory of this module,
and have an entry in `MODEL_REGISTRY`.

Each model class must have the following @property definitions
- `chunk_shape` (tuple[int, ...]): Shape of the tensor chunks to be sent to the sidecar
- `dtype` (torch.dtype): Data type of embeddings

Each model module must define the following classes
- `ModalityProcessor`: Inherited from `BaseModalityProcessor`, this class defines
    how to instantiate and run processors for each supported modality by overriding
    `get_image_processor`, `get_video_processor`, etc. The default implementation
    of these methods returns `None`, which is taken to mean that the modality is not
    supported by the model.
"""

import enum
from dataclasses import dataclass, field

from cornserve.task_executors.eric.schema import Modality


@dataclass
class WeightInfo:
    """Model info for a modality."""

    # Prefix of the model weights to collect
    prefix: str

    # Rules to replace weight name prefixes. For instance,
    # ("multi_modal.", "vision_tower.multi_modal.") will
    # find all weight names that start with "multi_modal.", strip
    # that prefix, and prepend with "vision_tower.multi_modal.".
    prefix_rename_rules: list[tuple[str, str]] = field(default_factory=list)


class ViTResolutionType(enum.Enum):
    """Resolution type of a ViT model."""

    # Fixed resolution ViT.
    # The patch size (e.g., 14x14) and resolution (e.g., 336x336) are fixed.
    # Many models will thus slice the input image into tiles with a fixed
    # resolution (number of patches) and batch them in ViT forward.
    FIXED = "fixed"

    # Dynamic resolution ViT.
    # The ViT can support virtually any number of patches. The input image
    # is sliced directly into patches and the whole sequence is passed to
    # the ViT.
    DYNAMIC = "dynamic"


@dataclass
class ModalityEntry:
    """Modality entry for a model class."""


@dataclass
class RegistryEntry:
    """Registry entry for a model class."""

    # Name of module within `models`
    module: str

    # Name of the model class
    class_name: str

    # Resolution type of the Vision Transformer model
    vit_resolution_type: ViTResolutionType

    # Modality to model info mapping
    weight: WeightInfo

    # Modality-specific info
    modality: dict[Modality, ModalityEntry]


# Keyed by a model's type (usually its HF config `model_type`).
MODEL_REGISTRY: dict[str, RegistryEntry] = {
    "qwen2_vl": RegistryEntry(
        module="qwen2_vl",
        class_name="Qwen2VisionTransformer",
        vit_resolution_type=ViTResolutionType.DYNAMIC,
        weight=WeightInfo(
            prefix="visual.",
        ),
        modality={
            Modality.IMAGE: ModalityEntry(),
            Modality.VIDEO: ModalityEntry(),
        },
    ),
    "llava_onevision": RegistryEntry(
        module="llava_onevision",
        class_name="LlavaOneVisionTransformer",
        vit_resolution_type=ViTResolutionType.FIXED,
        weight=WeightInfo(
            prefix="vision_tower.",
            prefix_rename_rules=[
                (
                    "multi_modal_projector.",
                    "vision_tower.multi_modal_projector.",
                ),
            ],
        ),
        modality={
            Modality.IMAGE: ModalityEntry(),
            Modality.VIDEO: ModalityEntry(),
        },
    ),
}
