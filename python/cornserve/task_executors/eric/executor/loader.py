import os
import json
import fnmatch
import hashlib
import filelock
import tempfile
import importlib
import contextlib
from typing import Literal, Type

import torch
import torch.nn as nn
import transformers
import safetensors
import huggingface_hub.errors
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoConfig, PretrainedConfig

from cornserve.logging import get_logger
from cornserve.task_executors.eric.schema import Modality
from cornserve.task_executors.eric.distributed import parallel
from cornserve.task_executors.eric.models import MODEL_REGISTRY

logger = get_logger(__name__)


def load_model(
    model_name_or_path: str,
    modality: Modality,
    weight_format: Literal["safetensors"] = "safetensors",
    cache_dir: str | None = None,
    revision: str | None = None,
    torch_dtype: torch.dtype | None = None,
    torch_device: torch.device | None = None,
) -> nn.Module:
    """Load a model from Hugging Face Hub.

    1. Instantiate the model.
    2. Download the model weights from Hugging Face Hub.
    3. Load the downloaded model weights into the model.

    Args:
        model_name_or_path: The model name or path.
        modality: The modality encoder to use from the model.
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
    try:
        model_class_name = registry_entry.model[modality].class_name
    except KeyError:
        logger.exception(
            "Modality %s not supported by %s. Available modalities in the registry are: %s",
            modality,
            model_name_or_path,
            registry_entry.model.keys(),
        )
        raise

    # Import the model class
    try:
        model_class: Type[nn.Module] = getattr(
            importlib.import_module(
                f"cornserve.task_executors.eric.models.{registry_entry.module}"
            ),
            model_class_name,
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
            "Model class %s not found in the `%s`. Registry entry: %s",
            model_class_name,
            f"models.{registry_entry.module}",
            registry_entry,
        )
        raise

    # Instantiate the model
    torch_dtype = torch_dtype or hf_config.torch_dtype
    assert isinstance(
        torch_dtype, torch.dtype
    ), f"torch_dtype is not a torch.dtype: {torch_dtype}"
    torch_device = torch_device or torch.device(
        "cuda", parallel.get_tensor_parallel_group().rank
    )
    with set_default_torch_dtype(torch_dtype), torch_device:
        model = model_class(hf_config)

    weight_dict = get_safetensors_weight_dict(
        model_name_or_path,
        weight_prefix=registry_entry.model[modality].weight_prefix,
        cache_dir=cache_dir,
        revision=revision,
    )

    incompatible = model.load_state_dict(weight_dict)
    if incompatible.missing_keys:
        logger.warning(
            "Some weights are missing in the model: %s",
            incompatible.missing_keys,
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Some weights are unexpected in the model: %s",
            incompatible.unexpected_keys,
        )

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
    weight_prefix: str = "",
    cache_dir: str | None = None,
    revision: str | None = None,
) -> dict[str, torch.Tensor]:
    """Download safetensors model weights from HF Hub and build weight dict.

    If possible, only download weights whose name starts with the prefix.
    The weight prefix will be stripped from the weight names in the dict.

    Args:
        model_name_or_path: The model name or path.
        weight_prefix: If possible, only download weights whose name starts with this.
        cache_dir: The cache directory to store the model weights. If None, will use HF defaults.
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
    with get_lock(model_name_or_path, cache_dir):
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
                if weight_name.startswith(weight_prefix):
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
    prefix_strip_len = len(weight_prefix)
    for weight_file in weight_files:
        with safetensors.safe_open(f"{hf_dir}/{weight_file}", framework="pt") as f:
            for name in f.keys():
                if name.startswith(weight_prefix):
                    weight_dict[name[prefix_strip_len:]] = f.get_tensor(name)
    return weight_dict


def get_lock(
    model_name_or_path: str, cache_dir: str | None = None
) -> filelock.BaseFileLock:
    """Get a file lock for the model directory."""
    lock_dir = cache_dir or tempfile.gettempdir()
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock
