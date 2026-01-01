"""Model registry for Geri."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cornserve.task_executors.geri.api import Modality
from cornserve.task_executors.geri.schema import GeriMode


@dataclass
class RegistryEntry:
    """Registry entry for a generation model class."""

    # Name of module within `models`
    module: str

    # Name of the model class
    class_name: str

    # Supported modalities for this model
    modalities: list[Modality]

    # Engine mode to run for this model
    geri_mode: GeriMode

    # Data type to run the model in
    torch_dtype: torch.dtype


# Keyed by HuggingFace model_index.json `_class_name` field
# or by `model_type` field in config.json
MODEL_REGISTRY: dict[str, RegistryEntry] = {
    "QwenImagePipeline": RegistryEntry(
        module="qwen_image",
        class_name="QwenImageModel",
        modalities=[Modality.IMAGE],
        geri_mode=GeriMode.BATCH,
        torch_dtype=torch.bfloat16,
    ),
    "qwen3_omni_moe": RegistryEntry(
        module="qwen3_omni_moe",
        class_name="Qwen3OmniMoeCode2Wav",
        modalities=[Modality.AUDIO],
        geri_mode=GeriMode.STREAMING,
        torch_dtype=torch.bfloat16,
    ),
}
