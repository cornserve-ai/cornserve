"""The registry holds model-specific information."""

import enum
from dataclasses import dataclass, field

from cornserve.task_executors.eric.schema import Modality


@dataclass
class WeightInfo:
    """Model info for a modality."""

    # List of model weight name prefixes to load.
    # Keys that start with any of these prefixes are downloaded, and they
    # will be included in the state dict loaded into the model.
    required_prefixes: list[str]

    # List of model weight name prefixes to ignore from the state dict.
    # Generally, you will add short prefixes to `required_prefixes` and
    # explicitly ignore specific longer submodules that we do not use.
    ignored_prefixes: list[str] = field(default_factory=list)

    # Whether or not to strip the prefixes from weight names before
    # calling `load_state_dict`.
    strip_prefixes: bool = True

    # Rules to replace weight name prefixes. For instance,
    # ("multi_modal.", "vision_tower.multi_modal.") will
    # find all weight names that start with "multi_modal.", strip
    # that prefix, and prepend with "vision_tower.multi_modal.".
    # prefix_rename_rules: list[tuple[str, str]] = field(default_factory=list)


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
            required_prefixes=["visual."],
            strip_prefixes=True,
        ),
        modality={
            Modality.IMAGE: ModalityEntry(),
            Modality.VIDEO: ModalityEntry(),
        },
    ),
    "llava_onevision": RegistryEntry(
        module="llava_onevision",
        class_name="LlavaOneVisionEncoder",
        vit_resolution_type=ViTResolutionType.FIXED,
        weight=WeightInfo(
            required_prefixes=["vision_tower.", "multi_modal_projector.", "image_newline"],
            ignored_prefixes=["vision_tower.vision_model.post_layernorm"],
            strip_prefixes=False,
        ),
        modality={
            Modality.IMAGE: ModalityEntry(),
            Modality.VIDEO: ModalityEntry(),
        },
    ),
}
