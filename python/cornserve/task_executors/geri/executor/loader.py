"""Model loading utilities for Geri."""

from __future__ import annotations

import importlib
import json

import torch
from huggingface_hub import hf_hub_download

from cornserve.logging import get_logger
from cornserve.task_executors.geri.models.base import GeriModel
from cornserve.task_executors.geri.models.registry import MODEL_REGISTRY

logger = get_logger(__name__)


def load_model(model_id: str, torch_device: torch.device) -> GeriModel:
    """Load a model for generation.

    Args:
        model_id: Hugging Face model ID.
        torch_device: Device to load the model on (e.g., torch.device("cuda"))

    Returns:
        Loaded model instance.

    Raises:
        KeyError: If the model type is not found in the registry.
        FileNotFoundError: If model_index.json is not found.
        ValueError: If the model configuration is invalid.
    """
    logger.info("Loading model %s", model_id)

    # Download and parse model_index.json to get the pipeline class name
    try:
        model_index_path = hf_hub_download(model_id, filename="model_index.json")
        with open(model_index_path) as f:
            model_index = json.load(f)

        pipeline_class_name = model_index["_class_name"]
        logger.info("Found pipeline class: %s", pipeline_class_name)

    except Exception as e:
        logger.exception("Failed to load model_index.json from %s", model_id)
        raise FileNotFoundError(f"Could not load model_index.json from {model_id}: {e}") from e

    # Get the registry entry for this pipeline class
    try:
        registry_entry = MODEL_REGISTRY[pipeline_class_name]
    except KeyError:
        logger.exception(
            "Pipeline class %s not found in registry. Available classes: %s",
            pipeline_class_name,
            list(MODEL_REGISTRY.keys()),
        )
        raise

    # Import the model class
    try:
        model_class: type[GeriModel] = getattr(
            importlib.import_module(f"cornserve.task_executors.geri.models.{registry_entry.module}"),
            registry_entry.class_name,
        )
    except ImportError:
        logger.exception(
            "Failed to import module `%s`. Registry entry: %s",
            registry_entry.module,
            registry_entry,
        )
        raise
    except AttributeError:
        logger.exception(
            "Model class %s not found in module `%s`. Registry entry: %s",
            registry_entry.class_name,
            f"models.{registry_entry.module}",
            registry_entry,
        )
        raise

    # Ensure that the model class is a GeriModel
    if not issubclass(model_class, GeriModel):
        raise ValueError(f"Model class {model_class} is not a subclass of GeriModel. Registry entry: {registry_entry}")

    # Instantiate the model
    model = model_class(model_id=model_id, torch_dtype=registry_entry.torch_dtype, torch_device=torch_device)

    logger.info("Model %s loaded successfully", model_id)
    return model
