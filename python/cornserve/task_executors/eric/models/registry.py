"""The registry holds model-specific information."""

from __future__ import annotations

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
    # Note that these prefixes are *after* prefix stripiing if you have
    # `strip_prefixes` set to `True`.
    ignored_prefixes: list[str] = field(default_factory=list)

    # List of model weight name prefixes that are to populate adapters.
    # Same as `ignored_prefixes` -- these prefixes are *after* prefix striping
    # if you have `strip_prefixes` set to `True`.
    adapter_prefixes: list[str] = field(default_factory=list)

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

    # Fully qualified name of the adapter submodule attribute.
    # Calling `getattr(model, adapter_attr)` should return the adapter
    adapter_attr: str | None

    # Resolution type of the Vision Transformer model
    vit_resolution_type: ViTResolutionType

    # Modality to model info mapping
    weight: WeightInfo

    # Modality-specific info
    modality: dict[Modality, ModalityEntry]

    def __post_init__(self) -> None:
        """Post-init hook to validate the registry entry."""
        if self.adapter_attr is not None and not self.weight.adapter_prefixes:
            raise ValueError(
                f"Registry entry for {self.class_name} must have `adapter_prefixes` set if `adapter_attr` is specified."
            )
        if self.adapter_attr is None and self.weight.adapter_prefixes:
            raise ValueError(
                f"Registry entry for {self.class_name} must have `adapter_attr` set if `adapter_prefixes` is specified."
            )


# Keyed by a model's type (usually its HF config `model_type`).
MODEL_REGISTRY: dict[str, RegistryEntry] = {
    "qwen2_vl": RegistryEntry(
        module="qwen2_vl",
        class_name="Qwen2VisionTransformer",
        adapter_attr=None,
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
    "qwen2_5_vl": RegistryEntry(
        module="qwen2_5_vl",
        class_name="Qwen2_5_VisionTransformer",
        adapter_attr=None,
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
    "qwen2_5_omni": RegistryEntry(
        module="qwen2_5_omni",
        class_name="Qwen2_5OmniEncoder",
        adapter_attr=None,
        vit_resolution_type=ViTResolutionType.DYNAMIC,
        weight=WeightInfo(
            required_prefixes=["thinker."],
            ignored_prefixes=["model.", "lm_head."],
            strip_prefixes=True,
        ),
        modality={
            Modality.IMAGE: ModalityEntry(),
            Modality.VIDEO: ModalityEntry(),
            Modality.AUDIO: ModalityEntry(),
        },
    ),
    "llava_onevision": RegistryEntry(
        module="llava_onevision",
        class_name="LlavaOneVisionEncoder",
        adapter_attr=None,
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
    "gemma3": RegistryEntry(
        module="gemma3",
        class_name="Gemma3VisionEncoder",
        adapter_attr="multi_modal_projector",
        vit_resolution_type=ViTResolutionType.FIXED,
        weight=WeightInfo(
            required_prefixes=["vision_tower.", "multi_modal_projector."],
            ignored_prefixes=["vision_tower.vision_model.post_layernorm"],
            adapter_prefixes=["multi_modal_projector."],
            strip_prefixes=False,
        ),
        modality={
            Modality.IMAGE: ModalityEntry(),
        },
    ),
}
