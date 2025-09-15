"""Utility functions to create cornserve app source code from templates."""

from pathlib import Path
from string import Template
from typing import Literal

MLLM_TEMPLATE_PATH = "apps/mllm.py.tmpl"
ERIC_TEMPLATE_PATH = "apps/eric.py.tmpl"
OMNI_TEMPLATE_PATH = "apps/omni.py.tmpl"
HF_OMNI_TEMPLATE_PATH = "apps/hf_omni.py.tmpl"
HF_IMAGE_TEMPLATE_PATH = "apps/hf_image.py.tmpl"


def create_mllm_app(
    model_id: str,
    task_class: Literal[
        "MLLMTask",
        "DisaggregatedMLLMTask",
        "NcclDisaggregatedMLLMTask",
    ] = "MLLMTask",
    modalities: list[Literal["IMAGE", "AUDIO", "VIDEO"]] = ["IMAGE"],
    encoder_fission: bool = True,
) -> str:
    """Create an MLLM app srouce code from a template.

    Args:
        model_id (str): The model identifier.
        task_class (str): The task class to be used.
        encoder_fission (str): Whether to use encoder fission, defaults to "False".
    """
    src = Path(MLLM_TEMPLATE_PATH).read_text()
    modalitiy_str = ", ".join([f"Modality.{m}" for m in modalities])
    rendered = Template(src).substitute(
        MODEL_ID=model_id,
        TASK_CLASS=task_class,
        MODALITIES=modalitiy_str,
        ENCODER_FISSION=str(encoder_fission),
    )
    return rendered.strip()


def create_eric_app(
    model_id: str,
    modality: Literal["IMAGE", "AUDIO", "VIDEO"] = "IMAGE",
    max_batch_size: int = 1,
) -> str:
    """Create an Eric app source code from a template.

    Args:
        model_id (str): The model identifier.
    """
    src = Path(ERIC_TEMPLATE_PATH).read_text()
    rendered = Template(src).substitute(
        MODEL_ID=model_id,
        MODALITY=f"Modality.{modality}",
        MAX_BATCH_SIZE=max_batch_size,
    )
    return rendered.strip()

def create_omni_app(
    model_id: str,
    modalities: list[Literal["IMAGE", "AUDIO", "VIDEO"]] = ["IMAGE"],
    encoder_fission: bool = True,
) -> str:
    """Create an Omni app source code from a template.

    Args:
        model_id (str): The model identifier.
    """
    if model_id != "Qwen/Qwen2.5-Omni-7B":
        raise ValueError("Only Qwen/Qwen2.5-Omni-7B is supported for Omni app.")
    src = Path(OMNI_TEMPLATE_PATH).read_text()
    modalitiy_str = ", ".join([f"Modality.{m}" for m in modalities])
    rendered = Template(src).substitute(
        MODALITIES=modalitiy_str,
        ENCODER_FISSION=str(encoder_fission),
    )
    return rendered.strip()

def create_hf_omni_app(
    model_id: str,
) -> str:
    """Create an Omni app source code from a template.

    Args:
        model_id (str): The model identifier.
    """
    if model_id != "Qwen/Qwen2.5-Omni-7B":
        raise ValueError("Only Qwen/Qwen2.5-Omni-7B is supported for Omni app.")
    src = Path(HF_OMNI_TEMPLATE_PATH).read_text()
    rendered = Template(src).substitute()
    return rendered.strip()

def create_hf_image_app(
    model_id: str,
) -> str:
    """Create an HF image app source code from a template.

    Args:
        model_id (str): The model identifier.
    """
    if model_id != "Qwen/Qwen-Image":
        raise ValueError("Only Qwen/Qwen-Image is supported for HF image app.")
    src = Path(HF_IMAGE_TEMPLATE_PATH).read_text()
    rendered = Template(src).substitute()
    return rendered.strip()
