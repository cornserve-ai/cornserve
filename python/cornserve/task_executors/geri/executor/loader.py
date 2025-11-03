"""Model loading utilities for Geri."""

from __future__ import annotations

import importlib
import json

import torch
from huggingface_hub import hf_hub_download
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from cornserve.logging import get_logger
from cornserve.task_executors.geri.models.base import GeriModel
from cornserve.task_executors.geri.models.registry import MODEL_REGISTRY, RegistryEntry

logger = get_logger(__name__)


def get_registry_entry(model_id: str) -> tuple[RegistryEntry, PretrainedConfig | None]:
    """Acquire the Geri model registry entry for the provided model_id.

    Args:
        model_id: Hugging Face model ID.

    Returns:
        Registry entry for the model and, if found, the HF model config.

    Raises:
        ValueError: If the model's class name not found in HF.
        KeyError: If the model type is not found in the registry.
    """
    logger.info("Looking up Geri model registry for model %s", model_id)

    class_name: str | None = None
    config: PretrainedConfig | None = None

    # First, try to download and parse model_index.json to get the pipeline class name
    try:
        model_index_path = hf_hub_download(model_id, filename="model_index.json")
        with open(model_index_path) as f:
            model_index = json.load(f)
        class_name = model_index["_class_name"]
        logger.info("Found pipeline class: %s", class_name)
    except Exception:
        logger.exception("Failed to load model_index.json from %s", model_id)

    # Then, if that didn't work, try to parse config.json for model_type instead
    if class_name is None:
        try:
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            class_name = config.model_type
            logger.info("Found model class: %s", class_name)
        except Exception:
            logger.exception("Failed to load config from %s", model_id)

    # Without the class name, it's not possible to find the registry entry
    if class_name is None:
        raise ValueError(f"Could not find class name from HF for model: {model_id}")

    # Get the registry entry for this class
    try:
        registry_entry = MODEL_REGISTRY[class_name]
    except KeyError:
        logger.exception(
            "Model class %s not found in registry. Available classes: %s",
            class_name,
            list(MODEL_REGISTRY.keys()),
        )
        raise

    return registry_entry, config


def load_model(
    model_id: str,
    torch_device: torch.device,
    registry_entry: RegistryEntry,
    config: PretrainedConfig | None = None,
) -> GeriModel:
    """Load a model for generation.

    Args:
        model_id: Hugging Face model ID.
        torch_device: Device to load the model on (e.g., torch.device("cuda"))
        registry_entry: The registry entry for this model.
        config: The config for this model.
            If supplied, the model will be initialized with the given config.
            If not supplied, the config for the model will be looked up if needed.

    Returns:
        Loaded model instance.

    Raises:
        ImportError: if the module specified in the registry entry could not be imported.
        AttributeError: if the class in specified in the registry entry does not exist in
            the module specified in the registry entry.
        ValueError: if the model is invalid.
    """
    logger.info("Loading model %s", model_id)

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
    model = model_class(
        model_id=model_id,
        torch_dtype=registry_entry.torch_dtype,
        torch_device=torch_device,
        config=config,
    )

    logger.info("Model %s loaded successfully", model_id)
    return model
