"""Utility functions to create cornserve app source code from templates."""

from pathlib import Path
from string import Template
from typing import Literal

MLLM_TEMPLATE_PATH = "apps/mllm.py.tmpl"
ERIC_TEMPLATE_PATH = "apps/eric.py.tmpl"


def create_mllm_app(
    model_id: str,
    task_class: Literal["MLLMTask", "DisaggregatedMLLMTask"] = "MLLMTask",
    encoder_fission: bool = True,
) -> str:
    """Create an MLLM app srouce code from a template.

    Args:
        model_id (str): The model identifier.
        task_class (str): The task class to be used.
        encoder_fission (str): Whether to use encoder fission, defaults to "False".
    """
    src = Path(MLLM_TEMPLATE_PATH).read_text()
    rendered = Template(src).substitute(MODEL_ID=model_id, TASK_CLASS=task_class, ENCODER_FISSION=str(encoder_fission))
    return rendered.strip()


def create_eric_app(
    model_id: str,
) -> str:
    """Create an Eric app source code from a template.

    Args:
        model_id (str): The model identifier.
    """
    src = Path(ERIC_TEMPLATE_PATH).read_text()
    rendered = Template(src).substitute(
        MODEL_ID=model_id,
    )
    return rendered.strip()
