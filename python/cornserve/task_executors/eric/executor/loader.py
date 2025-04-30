"""Instantiating the PyTorch model and loading Hugging Face Hub model weights."""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import importlib
import json
import os
import tempfile
from typing import Literal

import filelock
import huggingface_hub.errors
import safetensors
import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from cornserve.logging import get_logger
from cornserve.task_executors.eric.distributed import parallel
from cornserve.task_executors.eric.models.base import EricModel
from cornserve.task_executors.eric.models.registry import MODEL_REGISTRY

logger = get_logger(__name__)


def load_model(
    model_name_or_path: str,
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

    Args:
        model_name_or_path: The model name or path.
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

    # Ensure that the model class is an EricModel
    assert issubclass(model_class, EricModel), (
        f"Model class {model_class} is not a subclass of EricModel. Registry entry: {registry_entry}"
    )

    # Instantiate the model
    torch_dtype = torch_dtype or hf_config.torch_dtype
    assert isinstance(torch_dtype, torch.dtype), str(type(torch_dtype))
    torch_device = torch_device or torch.device("cuda", parallel.get_tensor_parallel_group().rank)
    with set_default_torch_dtype(torch_dtype), torch_device:
        model = model_class(hf_config)

    weight_dict = get_safetensors_weight_dict(
        model_name_or_path,
        weight_prefixes=registry_entry.weight.required_prefixes,
        strip_prefixes=registry_entry.weight.strip_prefixes,
        cache_dir=cache_dir,
        revision=revision,
    )

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
            raise ValueError(f"Unexpected weights in the model: {actually_unexpected_keys}")

    return model.eval()


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Context manager to set the default torch dtype."""
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def get_safetensors_weight_dict(
    model_name_or_path: str,
    weight_prefixes: list[str],
    strip_prefixes: bool,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> dict[str, torch.Tensor]:
    """Download safetensors model weights from HF Hub and build weight dict.

    If possible, only download weights whose name starts with the prefix.
    The weight prefix will be stripped from the weight names in the dict.

    Args:
        model_name_or_path: The model name or path.
        weight_prefixes: Only download weights whose names starts with these.
            If the repo does not have a weight index file, all weights will be
            downloaded regardless of the prefix.
        strip_prefixes: Whether to strip the prefixes from weight names before
            collecting weights into the dict.
        cache_dir: The cache directory to store the model weights.
            If None, will use HF defaults.
        revision: The revision of the model.

    Returns:
        A dictionary mapping weight names to tensors.
    """
    # Select the first pattern with matching files on the hub
    file_list = HfApi().list_repo_files(model_name_or_path, revision=revision)
    weight_files = fnmatch.filter(file_list, "*.safetensors")
    if not weight_files:
        raise FileNotFoundError("No .safetensors files found in the model repo.")

    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with with_lock(model_name_or_path, cache_dir):
        # Try to filter the list of safetensors files using the index and prefix.
        # Note that not all repositories have an index file.
        try:
            index_file_path = hf_hub_download(
                model_name_or_path,
                filename=transformers.utils.SAFE_WEIGHTS_INDEX_NAME,
                cache_dir=cache_dir,
                revision=revision,
            )
            # Maps weight names to the safetensors file they are stored in
            with open(index_file_path) as f:
                weight_map = json.load(f)["weight_map"]
            weight_files = []
            for weight_name, weight_file in weight_map.items():
                # Only keep weights that start with the prefix
                if any(weight_name.startswith(p) for p in weight_prefixes):
                    weight_files.append(weight_file)
                    break
            logger.info(
                "Safetensors file to download (filtered by index): %s",
                weight_files,
            )
        except huggingface_hub.errors.EntryNotFoundError:
            logger.info(
                "No safetensors index file found. Downloading all .safetensors files: %s",
                weight_files,
            )

        # Download the safetensors files
        hf_dir = snapshot_download(
            model_name_or_path,
            allow_patterns=weight_files,
            cache_dir=cache_dir,
            revision=revision,
        )

    # Build weight dict
    weight_dict = {}
    prefix_lens = [len(p) for p in weight_prefixes]
    for weight_file in weight_files:
        with safetensors.safe_open(f"{hf_dir}/{weight_file}", framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                for weight_prefix, strip_len in zip(weight_prefixes, prefix_lens, strict=True):
                    if name.startswith(weight_prefix):
                        stripped_name = name[strip_len:] if strip_prefixes else name
                        weight_dict[stripped_name] = f.get_tensor(name)
                        break

    return weight_dict


@contextlib.contextmanager
def with_lock(model_name_or_path: str, cache_dir: str | None = None):
    """Get a file lock for the model directory."""
    lock_dir = cache_dir or tempfile.gettempdir()
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    lock.acquire()
    yield
    lock.release()
    # Clean up the lock file
    with contextlib.suppress(FileNotFoundError):
        os.remove(lock.lock_file)
