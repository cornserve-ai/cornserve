"""Instantiating the PyTorch model and loading Hugging Face Hub model weights."""

from __future__ import annotations

import importlib
from typing import Literal

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from cornserve.logging import get_logger
from cornserve.task_executors.eric.distributed import parallel
from cornserve.task_executors.eric.models.base import BaseAdapterModule, EricModel
from cornserve.task_executors.eric.models.registry import MODEL_REGISTRY
from cornserve.task_executors.utils import get_safetensors_weight_dict, set_default_torch_dtype

logger = get_logger(__name__)


def load_model(
    model_name_or_path: str,
    adapter_model_names_or_paths: list[str] | None = None,
    weight_format: Literal["safetensors"] = "safetensors",
    cache_dir: str | None = None,
    revision: str | None = None,
    torch_dtype: torch.dtype | None = None,
    torch_device: torch.device | None = None,
) -> EricModel:
    """Load a model from Hugging Face Hub.

    1. Instantiate the model.
    2. Download the model weights from Hugging Face Hub.
    3. Load the downloaded model weights into the model.
    4. If specified by the registry and the model config, load adapters into the model.

    Args:
        model_name_or_path: The model name or path.
        adapter_model_names_or_paths: Optional list of model names or paths to load adapters from.
        weight_format: The format of the model weights. Currently only "safetensors" is supported.
        cache_dir: The cache directory to store the model weights. If None, will use HF defaults.
        revision: The revision of the model.
        torch_dtype: The torch dtype to use. If None, will use the dtype from the model config.
        torch_device: The torch device to use. If None, will use CUDA and current TP rank.

    Returns:
        A PyTorch nn.Module instance.
    """
    if weight_format not in ["safetensors"]:
        raise ValueError("Only 'safetensors' format is supported.")

    # Fetch the model config from HF
    hf_config: PretrainedConfig = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
        trust_remote_code=True,
    )

    # Fetch the model class name from the registry
    try:
        registry_entry = MODEL_REGISTRY[hf_config.model_type]
    except KeyError:
        logger.exception(
            "Model type %s not found in the registry. Available model types are: %s",
            model_name_or_path,
            MODEL_REGISTRY.keys(),
        )
        raise

    # Import the model class
    try:
        model_class: type[EricModel] = getattr(
            importlib.import_module(f"cornserve.task_executors.eric.models.{registry_entry.module}"),
            registry_entry.class_name,
        )
    except ImportError:
        logger.exception(
            "Failed to import `%s` from `models`. Registry entry: %s",
            registry_entry.module,
            registry_entry,
        )
        raise
    except AttributeError:
        logger.exception(
            "Model class %s not found in `%s`. Registry entry: %s",
            registry_entry.class_name,
            f"models.{registry_entry.module}",
            registry_entry,
        )
        raise

    # Ensure that the model class is an EricModel.
    assert issubclass(model_class, EricModel), (
        f"Model class {model_class} is not a subclass of EricModel. Registry entry: {registry_entry}"
    )

    # Determine dtype and device.
    if torch_dtype is None:
        hf_dtype = hf_config.dtype or hf_config.text_config.dtype
        if not isinstance(hf_dtype, torch.dtype):
            raise ValueError(
                f"Expected torch_dtype to be a torch.dtype, but got {type(hf_dtype)}. "
                "Please specify torch_dtype explicitly."
            )
        torch_dtype = hf_dtype
    assert isinstance(torch_dtype, torch.dtype), str(type(torch_dtype))
    torch_device = torch_device or torch.device("cuda", parallel.get_tensor_parallel_group().rank)

    # Instantiate the model
    with set_default_torch_dtype(torch_dtype), torch_device:
        model = model_class(hf_config)

    # Download weights, but only the ones we actually need.
    weight_dict = get_safetensors_weight_dict(
        model_name_or_path,
        weight_prefixes=registry_entry.weight.required_prefixes,
        strip_prefixes=registry_entry.weight.strip_prefixes,
        cache_dir=cache_dir,
        revision=revision,
    )

    # Exclude prefixes specified as for adapters. They'll be loaded separately.
    for prefix in registry_entry.weight.adapter_prefixes:
        weight_dict = {k: v for k, v in weight_dict.items() if not k.startswith(prefix)}

    # Load model weights.
    incompatible = model.load_state_dict(weight_dict, strict=False)
    if incompatible.missing_keys:
        raise ValueError(f"Missing weights in the model: {incompatible.missing_keys}")
    if keys := incompatible.unexpected_keys:
        # Some keys in the state dict are explicitly ignored since we dont' use them.
        actually_unexpected_keys = []
        for key in keys:
            if not any(key.startswith(prefix) for prefix in registry_entry.weight.ignored_prefixes):
                actually_unexpected_keys.append(key)
        if actually_unexpected_keys:
            raise ValueError(
                f"Unexpected weights in the model: {actually_unexpected_keys}.\n\n"
                f"Expected weights: {[name for name, _ in model.named_parameters()]}"
            )

    # If there are no adapters for this module, return early.
    if not registry_entry.weight.adapter_prefixes:
        logger.info("No adapters to load for model %s", model_name_or_path)
        return model.eval()

    # Fish out the adapter submodule and load adapter weight into it.
    assert registry_entry.adapter_attr is not None
    adapter = getattr(model, registry_entry.adapter_attr)
    assert isinstance(adapter, BaseAdapterModule), (
        f"Adapter submodule {registry_entry.adapter_attr} is not a BaseAdapterModule. Got {type(adapter)}."
    )
    for model_id in [model_name_or_path, *(adapter_model_names_or_paths or [])]:
        logger.info("Loading adapter weights from %s", model_id)

        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, revision=revision)

        with set_default_torch_dtype(torch_dtype), torch_device:
            new_adapter_module = adapter.make_adapter(model_id, config)

        adapter_weight_dict = get_safetensors_weight_dict(
            model_id,
            weight_prefixes=registry_entry.weight.adapter_prefixes,
            strip_prefixes=True,
            cache_dir=cache_dir,
            revision=revision,
        )

        incompatible = new_adapter_module.load_state_dict(adapter_weight_dict, strict=False)
        if incompatible.missing_keys:
            raise ValueError(f"Missing adapter weights: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected adapter weights: {incompatible.unexpected_keys}")

    return model.eval()
