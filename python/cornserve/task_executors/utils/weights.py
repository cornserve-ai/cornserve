"""Utilities for weight loading logic used across task executors."""

import contextlib
import fnmatch
import hashlib
import json
import os
import tempfile

import filelock
import huggingface_hub
import safetensors
import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from cornserve.logging import get_logger

logger = get_logger(__name__)


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
            weight_files_set: set[str] = set()
            for weight_name, weight_file in weight_map.items():
                # Only keep weights that start with the prefix
                if any(weight_name.startswith(p) for p in weight_prefixes):
                    weight_files_set.add(weight_file)
            weight_files = sorted(weight_files_set)
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
