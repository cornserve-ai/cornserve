"""Dataset classes for benchmark workload generation.

This module provides dataset types for different model categories:
  - VLMRequest/ControlledImageChat: Vision-language models with image+text input
  - OmniRequest/ServeGenDataset: Omni-modal models with image/video/audio+text input
  - DiTRequest/ControlledDiT: Diffusion transformer image generation
"""

import os
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import random
from typing import Any

import datasets as hf_datasets
import numpy as np
from pydantic import BaseModel
from servegen import Category, ClientPool
from servegen.construct import generate_workload
from servegen.utils import get_scaled_rate_fn
from transformers import AutoTokenizer

from mm_utils import (
    create_dummy_audio,
    create_dummy_image,
    create_dummy_video,
    get_file_data_uris,
)
from servegen_distributions import add_update_multimodal_content

logger = logging.getLogger(__name__)

REQUESTS_DIR = "requests"


# -----------------------------------------------------------------------------
# Dataset Base Class
# -----------------------------------------------------------------------------


class Dataset(BaseModel):
    """Base class for all datasets."""

    @abstractmethod
    def to_suffix(self) -> str:
        """Return a string suffix describing the dataset configuration.

        This is used to generate unique filenames for benchmark results.
        """
        ...

    @abstractmethod
    def sample(self, model_id: str) -> list[Any]:
        """Sample requests from the dataset.

        Args:
            model_id: HuggingFace model ID to use for tokenization.

        Returns:
            List of request objects (VLMRequest, OmniRequest, or DiTRequest).
        """
        ...


# -----------------------------------------------------------------------------
# Request Persistence Helpers
# -----------------------------------------------------------------------------


def with_saved_requests(request_type: str, request_class: type[Any]):
    """Decorator to persist sampled requests to disk and reload on repeated runs.

    Args:
        request_type: Type of request ("VLM", "Omni", or "DiT").
        request_class: Request class to instantiate when loading from disk.
    """

    def decorator(sample_func):
        def wrapper(self, model_id: str):
            path = get_request_path(request_type, model_id, self.to_suffix())

            if path.exists():
                logger.info(f"Loading {request_type} requests from {path}")
                return load_requests(path, request_class)

            logger.info(
                f"No saved requests found, sampling new {request_type} requests"
            )
            requests = sample_func(self, model_id)
            save_requests(requests, path)

            return requests

        return wrapper

    return decorator


def setup_sampling(model_id: str, random_seed: int) -> Any:
    """Common setup for sampling: set seed and create tokenizer.

    Args:
        model_id: HuggingFace model ID.
        random_seed: Random seed for reproducibility.

    Returns:
        Tokenizer instance.
    """
    np.random.seed(random_seed)
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _ensure_dummy_file(modality: str, filename: str) -> None:
    """Recreate a dummy media file if it's missing (e.g. after cache restore).

    Parses the deterministic filename to recover creation parameters:
      image: {height}x{width}_{id}.png
      video: {num_frames}x{height}x{width}_{id}.mp4
      audio: {duration_sec}s_{id}.wav
    """
    root = f"mm_data/{modality}s"
    if os.path.exists(os.path.join(root, filename)):
        return
    stem = filename.rsplit(".", 1)[0]  # strip extension
    name, id_str = stem.rsplit("_", 1)
    file_id = int(id_str)
    if modality == "image":
        h, w = map(int, name.split("x"))
        create_dummy_image(height=h, width=w, id=file_id)
    elif modality == "video":
        parts = name.split("x")
        create_dummy_video(
            num_frames=int(parts[0]), height=int(parts[1]),
            width=int(parts[2]), id=file_id,
        )
    elif modality == "audio":
        dur = int(name.rstrip("s"))
        create_dummy_audio(duration_sec=dur, id=file_id)


def materialize_mm_data_from_filenames(
    filenames: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Materialize multi_modal_data_list from filenames using mm_utils.

    Recreates any missing dummy media files before reading them.

    Args:
        filenames: List of (modality, filename) tuples.

    Returns:
        List of multimodal data dictionaries ready for API requests.
    """
    mm_data = []
    for modality, filename in filenames:
        _ensure_dummy_file(modality, filename)
        if modality == "image":
            uri = get_file_data_uris([filename], root="mm_data/images")[0]
            mm_data.append({"type": "image_url", "image_url": {"url": uri}})
        elif modality == "video":
            uri = get_file_data_uris([filename], root="mm_data/videos")[0]
            mm_data.append({"type": "video_url", "video_url": {"url": uri}})
        elif modality == "audio":
            uri = get_file_data_uris([filename], root="mm_data/audios")[0]
            mm_data.append({"type": "audio_url", "audio_url": {"url": uri}})
    return mm_data


def create_mm_entry(
    modality: str, filename: str
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Create a multimodal entry (filename tuple and data dict).

    Args:
        modality: Modality type ("image", "video", or "audio").
        filename: Filename to use.

    Returns:
        Tuple of (filename_tuple, mm_data_dict).
    """
    filename_tuple = (modality, filename)
    uri = get_file_data_uris([filename], root=f"mm_data/{modality}s")[0]
    mm_data_dict = {"type": f"{modality}_url", f"{modality}_url": {"url": uri}}
    return filename_tuple, mm_data_dict


def get_request_path(request_type: str, model_id: str, suffix: str) -> Path:
    """Get the file path for storing sampled requests.

    Args:
        request_type: Type of request ("VLM", "Omni", or "DiT").
        model_id: HuggingFace model ID.
        suffix: Dataset suffix from to_suffix().

    Returns:
        Path to the request file.
    """
    safe_model_id = model_id.replace("/", "_")
    request_dir = Path(REQUESTS_DIR) / request_type / safe_model_id
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir / f"{suffix}.json"


def save_requests(
    requests: list[Any],
    path: Path,
) -> None:
    """Save requests to disk without multi_modal_data_list.

    Args:
        requests: List of request objects.
        path: Path to save the request file.
    """
    data = []
    for req in requests:
        req_dict = asdict(req)
        if "multi_modal_data_list" in req_dict:
            del req_dict["multi_modal_data_list"]
        data.append(req_dict)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(requests)} requests to {path}")


def load_requests(
    path: Path,
    request_class: type[Any],
) -> list[Any]:
    """Load requests from disk and materialize multi_modal_data_list.

    Args:
        path: Path to the request file.
        request_class: Class to instantiate (VLMRequest, OmniRequest, or DiTRequest).

    Returns:
        List of request objects with materialized multi_modal_data_list.
    """
    with open(path, "r") as f:
        data = json.load(f)

    requests = []
    for req_dict in data:
        if "filenames" in req_dict:
            filenames = (
                [tuple(f) for f in req_dict["filenames"]]
                if req_dict["filenames"]
                else []
            )
            req_dict["multi_modal_data_list"] = materialize_mm_data_from_filenames(
                filenames
            )

        requests.append(request_class(**req_dict))

    logger.info(f"Loaded {len(requests)} requests from {path}")
    return requests  # type: ignore


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class VLMRequest:
    """Represents a vision-language model request with image+text input."""

    prompt: str | Any
    prompt_len: int
    output_len: int
    multi_modal_data_list: list[dict[str, Any]]

    # mm data bookkeeping: list of (modality, filename) tuples
    filenames: list[tuple[str, str]] = field(default_factory=list)

    # servegen timestamp (optional, used when sampled from ServGen)
    timestamp: float | None = None

    # total multimodal encoder tokens (images + videos + audio)
    mm_encoder_tokens: int = 0


@dataclass
class OmniRequest:
    """Represents an omni-modal model request with image/video/audio+text input."""

    prompt: str | Any
    prompt_len: int
    output_len: int
    multi_modal_data_list: list[dict[str, Any]]

    # mm data bookkeeping: list of (modality, filename) tuples
    filenames: list[tuple[str, str]] = field(default_factory=list)

    # servegen timestamp
    timestamp: float | None = None

    # omni-specific: whether to return audio output
    return_audio: bool = False

    # total multimodal encoder tokens (images + videos + audio)
    mm_encoder_tokens: int = 0


@dataclass
class DiTRequest:
    """Represents a diffusion transformer (DiT) image generation request."""

    prompt: str | Any
    prompt_len: int
    output_image_height: int
    output_image_width: int
    num_inference_steps: int


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def generate_random_text_with_length(
    tokenizer: Any,
    target_length: int,
) -> str:
    """Generate random text that tokenizes to approximately target_length tokens.

    Args:
        tokenizer: Tokenizer to use for measuring token length.
        target_length: Target number of tokens.

    Returns:
        Random text string that tokenizes to approximately target_length tokens.
    """
    if target_length <= 0:
        return ""

    special_ids = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)

    # Pre-generate a long sequence of random token IDs
    # hard coded max len
    pool_size = max(32768, target_length * 2)
    long_token_ids = []

    while len(long_token_ids) < pool_size:
        # Generate extra to account for filtering
        batch_size = pool_size + max(50, pool_size // 100)
        random_tokens = np.random.randint(
            0, vocab_size, size=batch_size, dtype=np.int32
        )

        # Filter out special tokens
        if special_ids:
            mask = ~np.isin(random_tokens, list(special_ids))
            valid_tokens = random_tokens[mask]
        else:
            valid_tokens = random_tokens

        long_token_ids.extend(valid_tokens.tolist())

        if len(long_token_ids) >= pool_size:
            long_token_ids = long_token_ids[:pool_size]
            break

    # Take a slightly longer slice than needed (add some buffer)
    buffer_size = min(
        50, max(10, target_length // 10)
    )  # At least 10, up to 50 tokens buffer
    slice_length = min(target_length + buffer_size, len(long_token_ids))
    token_slice = long_token_ids[:slice_length]

    random.shuffle(token_slice)

    # Fine-tune to get exact target_length after tokenization
    current_tokens = token_slice[:target_length]
    prompt = tokenizer.decode(current_tokens, clean_up_tokenization_spaces=True).strip()

    encoded_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    actual_length = len(encoded_tokens) + 1  # for the eos token
    current_tokens = encoded_tokens

    if actual_length == target_length:
        return prompt
    elif actual_length < target_length:
        # Need more tokens, add from our shuffled pool
        needed = target_length - actual_length
        if len(current_tokens) + needed <= len(token_slice):
            current_tokens = token_slice[: len(current_tokens) + needed]
        else:
            needed_from_pool = needed - (slice_length - target_length)
            current_tokens = (
                token_slice
                + long_token_ids[slice_length : slice_length + needed_from_pool]
            )
    else:
        # Too many tokens, remove some
        excess = actual_length - target_length
        current_tokens = current_tokens[: max(1, len(current_tokens) - excess)]

    prompt = tokenizer.decode(current_tokens, clean_up_tokenization_spaces=True).strip()
    return prompt


# -----------------------------------------------------------------------------
# Dataset Classes
# -----------------------------------------------------------------------------


class ControlledImageChat(Dataset):
    """Generate controlled image+text chat requests with specified characteristics."""

    num_requests: int
    input_len: int
    output_len: int
    image_width: int
    image_height: int
    image_count: int = 1
    image_probability: float = 1.0
    random_seed: int = 0

    def to_suffix(self) -> str:
        suffix = (
            f"controlled_image_chat+requests{self.num_requests}+"
            f"input{self.input_len}+output{self.output_len}+"
            f"img_w{self.image_width}_h{self.image_height}+"
            f"count{self.image_count}+prob{self.image_probability}"
        )
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    @with_saved_requests("VLM", VLMRequest)
    def sample(
        self,
        model_id: str,
    ) -> list[VLMRequest]:
        """Generate controlled image chat requests.

        Args:
            model_id: HuggingFace model ID to use for tokenization.

        Returns:
            List of VLMRequest objects.
        """
        tokenizer = setup_sampling(model_id, self.random_seed)
        sampled_requests: list[VLMRequest] = []

        # Create a single dummy image (reused for all requests)
        image_filename = create_dummy_image(
            width=self.image_width, height=self.image_height, id=0
        )

        for _ in range(self.num_requests):
            prompt = generate_random_text_with_length(tokenizer, self.input_len)
            mm_data = []
            filenames = []

            # Include images based on probability
            if np.random.rand() < self.image_probability:
                for _ in range(self.image_count):
                    filename_tuple, mm_data_dict = create_mm_entry(
                        "image", image_filename
                    )
                    filenames.append(filename_tuple)
                    mm_data.append(mm_data_dict)

            sampled_requests.append(
                VLMRequest(
                    prompt=prompt,
                    prompt_len=len(tokenizer(prompt).input_ids),
                    multi_modal_data_list=mm_data,
                    output_len=self.output_len,
                    filenames=filenames,
                )
            )

        return sampled_requests


class PrefillLLMDataset(Dataset):
    """Text-only requests for prefill profiling; all requests have output_len=1."""

    num_requests: int
    input_len: int
    random_seed: int = 0

    def to_suffix(self) -> str:
        suffix = f"prefill+requests{self.num_requests}+input{self.input_len}"
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    @with_saved_requests("VLM", VLMRequest)
    def sample(self, model_id: str) -> list[VLMRequest]:
        tokenizer = setup_sampling(model_id, self.random_seed)
        return [
            VLMRequest(
                prompt=generate_random_text_with_length(tokenizer, self.input_len),
                prompt_len=self.input_len,
                output_len=1,
                multi_modal_data_list=[],
                filenames=[],
            )
            for _ in range(self.num_requests)
        ]


class DecodeLLMDataset(Dataset):
    """All requests share one fixed prefix for decode profiling via prefix caching."""

    num_requests: int
    output_len: int
    prefix_len: int = 2048
    random_seed: int = 0

    def to_suffix(self) -> str:
        suffix = f"decode+prefix{self.prefix_len}+out{self.output_len}+n{self.num_requests}"
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    @with_saved_requests("VLM", VLMRequest)
    def sample(self, model_id: str) -> list[VLMRequest]:
        tokenizer = setup_sampling(model_id, self.random_seed)
        prefix = generate_random_text_with_length(tokenizer, self.prefix_len)
        actual_prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
        return [
            VLMRequest(
                prompt=prefix,
                prompt_len=actual_prefix_len,
                output_len=self.output_len,
                multi_modal_data_list=[],
                filenames=[],
            )
            for _ in range(self.num_requests)
        ]


class DecodeServeGenDataset(Dataset):
    """Decode profiling with servegen-sampled output lengths and timestamps.

    All requests share a single text prefix (for prefix caching), but each
    request has its own output_len and timestamp drawn from the servegen
    distribution.  The prefix_len should account for visual tokens when
    simulating image-heavy workloads (text + image_prob * avg_images * visual_tokens).
    """

    request_rate: float
    duration: int
    image_prob: float
    prefix_len: int
    random_seed: int = 0

    def to_suffix(self) -> str:
        suffix = (
            f"decode_servegen+rate{self.request_rate}+duration{self.duration}"
            f"+img_prob{self.image_prob}+prefix{self.prefix_len}"
        )
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    @with_saved_requests("VLM", VLMRequest)
    def sample(self, model_id: str) -> list[VLMRequest]:
        tokenizer = setup_sampling(model_id, self.random_seed)

        # Sample from servegen to get realistic output_len + timestamps
        pool = ClientPool(
            Category.MULTIMODAL, "mm-image", data_dir="../third_party/servegen/data"
        )
        view = pool.span(0, float("inf"))
        rate_fn = get_scaled_rate_fn(view, self.request_rate / 5.55)
        requests = generate_workload(
            view, rate_fn, duration=self.duration, seed=self.random_seed
        )

        # Build shared prefix
        prefix = generate_random_text_with_length(tokenizer, self.prefix_len)
        actual_prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))

        return [
            VLMRequest(
                prompt=prefix,
                prompt_len=actual_prefix_len,
                output_len=int(req.data["output_tokens"]),
                multi_modal_data_list=[],
                filenames=[],
                timestamp=req.timestamp,
            )
            for req in requests
        ]


class DiffusionDBTextChat(Dataset):
    """Generate text-only chat requests using prompts sampled from DiffusionDB."""

    num_requests: int
    output_len: int
    random_seed: int = 0

    def to_suffix(self) -> str:
        suffix = (
            f"diffusiondb_text_chat+requests{self.num_requests}+output{self.output_len}"
        )
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    @with_saved_requests("VLM", VLMRequest)
    def sample(
        self,
        model_id: str,
    ) -> list[VLMRequest]:
        """Generate text-only requests from DiffusionDB prompts.

        Args:
            model_id: HuggingFace model ID to use for tokenization.

        Returns:
            List of VLMRequest objects without multimodal inputs.
        """

        tokenizer = setup_sampling(model_id, self.random_seed)

        ds = hf_datasets.load_dataset(
            "parquet",
            data_files="hf://datasets/poloclub/diffusiondb/metadata.parquet",
            split="train",
        )
        ds = ds.shuffle(seed=self.random_seed)

        prompts: list[str] = []
        while len(prompts) < self.num_requests:
            remaining = self.num_requests - len(prompts)
            prompts.extend(ds["prompt"][:remaining])

        return [
            VLMRequest(
                prompt=prompt,
                prompt_len=len(tokenizer(prompt).input_ids),
                output_len=self.output_len,
                multi_modal_data_list=[],
                filenames=[],
            )
            for prompt in prompts
        ]


class ControlledDiT(Dataset):
    """Generate controlled DiT / image generation requests with resolution sampling.

    Prompts are sampled from DiffusionDB, giving each request a natural, varying
    prompt length.  Resolution sampling is controlled via ``resolution_pool`` and
    ``resolution_weights``.
    """

    num_requests: int
    num_inference_steps: int
    resolution_pool: list[tuple[int, int]]
    resolution_weights: list[float] | None = None
    tokenizer_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    random_seed: int = 0

    def to_suffix(self) -> str:
        resolutions = "_".join(f"{h}x{w}" for h, w in self.resolution_pool)
        suffix = (
            f"controlled_dit+requests{self.num_requests}+"
            f"diffusiondb+steps{self.num_inference_steps}+"
            f"resolutions{resolutions}"
        )
        if self.resolution_weights is not None:
            weights = "_".join(f"{w:.2f}" for w in self.resolution_weights)
            suffix += f"+weights{weights}"
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    @with_saved_requests("DiT", DiTRequest)
    def sample(
        self,
        model_id: str,
    ) -> list[DiTRequest]:
        """Generate controlled DiT image generation requests.

        Prompts are drawn from DiffusionDB so that each request has a
        realistic, varying prompt length.  Resolutions are sampled from
        ``resolution_pool`` according to ``resolution_weights``.

        Args:
            model_id: HuggingFace model ID to use for tokenization.

        Returns:
            List of DiTRequest objects.
        """

        tokenizer = setup_sampling(self.tokenizer_model_id, self.random_seed)
        sampled_requests: list[DiTRequest] = []

        # --- Load & sample prompts from DiffusionDB ---
        # Load the metadata parquet directly (compatible with datasets >= 4.0).
        ds = hf_datasets.load_dataset(
            "parquet",
            data_files="hf://datasets/poloclub/diffusiondb/metadata.parquet",
            split="train",
        )
        ds = ds.shuffle(seed=self.random_seed)
        # Cycle if num_requests > dataset size
        prompts: list[str] = []
        while len(prompts) < self.num_requests:
            remaining = self.num_requests - len(prompts)
            prompts.extend(ds["prompt"][:remaining])

        # --- Normalize resolution weights ---
        if self.resolution_weights is None:
            weights = [1.0 / len(self.resolution_pool)] * len(self.resolution_pool)
        else:
            total = sum(self.resolution_weights)
            weights = [w / total for w in self.resolution_weights]

        if len(self.resolution_pool) != len(weights):
            raise ValueError(
                "resolution_pool and resolution_weights must have the same length"
            )

        # --- Build requests ---
        for prompt in prompts:
            idx = np.random.choice(len(self.resolution_pool), p=weights)
            height, width = self.resolution_pool[idx]

            sampled_requests.append(
                DiTRequest(
                    prompt=prompt,
                    prompt_len=len(tokenizer(prompt).input_ids),
                    output_image_height=height,
                    output_image_width=width,
                    num_inference_steps=self.num_inference_steps,
                )
            )

        return sampled_requests


class ServeGenDataset(Dataset):
    """Generate realistic multimodal workloads using ServGen library."""

    request_rate: float
    duration: int
    image_prob: float
    audio_prob: float = 0.0
    video_prob: float = 0.0
    return_audio_prob: float = 0.0
    vlm_only: bool = False
    random_seed: int = 0

    def to_suffix(self) -> str:
        """Return the dataset configuration as a string suffix."""
        suffix = (
            f"servegen+rate{self.request_rate}+duration{self.duration}+"
            f"img_prob{self.image_prob}+audio_prob{self.audio_prob}+"
            f"video_prob{self.video_prob}"
        )
        if self.return_audio_prob > 0.0:
            suffix += f"+return_audio_prob{self.return_audio_prob}"
        if self.vlm_only:
            suffix += "+vlm_only"
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    def sample(
        self,
        model_id: str,
    ) -> list[VLMRequest] | list[OmniRequest]:
        """Generate multimodal workload using ServGen.

        Args:
            model_id: HuggingFace model ID to use for tokenization.

        Returns:
            List of VLMRequest (if vlm_only=True) or OmniRequest (if vlm_only=False) objects with timestamps.
        """
        request_type = "VLM" if self.vlm_only else "Omni"
        path = get_request_path(request_type, model_id, self.to_suffix())

        if path.exists():
            logger.info(f"Loading {request_type} requests from {path}")
            request_class = VLMRequest if self.vlm_only else OmniRequest
            return load_requests(path, request_class)

        logger.info(f"No saved requests found, sampling new {request_type} requests")
        print("\n=== Generating Multimodal Workload ===")
        tokenizer = setup_sampling(model_id, self.random_seed)

        # Load image workload from ServGen
        pool = ClientPool(
            Category.MULTIMODAL, "mm-image", data_dir="../third_party/servegen/data"
        )
        print(f"Loaded {len(pool.clients)} multimodal clients")
        view = pool.span(0, float("inf"))

        rate_fn = get_scaled_rate_fn(view, self.request_rate / 5.55)

        requests = generate_workload(
            view, rate_fn, duration=self.duration, seed=self.random_seed
        )

        # Add video and audio data sampled from empirically fit distributions
        # based on video_prob and audio_prob. Remove images based on image_prob.
        # When vlm_only=True, force video_prob and audio_prob to 0
        requests = add_update_multimodal_content(
            requests,
            no_image_prob=1.0 - self.image_prob,
            video_prob=0.0 if self.vlm_only else self.video_prob,
            audio_prob=0.0 if self.vlm_only else self.audio_prob,
            seed=self.random_seed,
        )

        # Print statistics
        print("\nMultimodal Workload Statistics:")
        print(f"Total Requests: {len(requests)}")
        print(
            f"Time Range: {requests[0].timestamp:.2f} to {requests[-1].timestamp:.2f}"
        )

        # Calculate average token counts
        avg_text = sum(r.data["text_tokens"] for r in requests) / len(requests)  # type: ignore
        avg_output = sum(r.data["output_tokens"] for r in requests) / len(requests)  # type: ignore
        avg_image = sum(len(r.data["image_tokens"]) for r in requests) / len(requests)  # type: ignore
        avg_audio = sum(len(r.data["audio_tokens"]) for r in requests) / len(requests)  # type: ignore
        avg_video = sum(len(r.data["video_tokens"]) for r in requests) / len(requests)  # type: ignore

        print("\nAverage Token Counts:")
        print(f"  Text tokens: {avg_text:.2f}")
        print(f"  Output tokens: {avg_output:.2f}")
        print(f"  Image count: {avg_image:.2f}")
        print(f"  Audio count: {avg_audio:.2f}")
        print(f"  Video count: {avg_video:.2f}")

        sampled_requests: list[VLMRequest] | list[OmniRequest] = []

        for req in requests:
            input_len = req.data["text_tokens"]
            prompt = generate_random_text_with_length(tokenizer, input_len)  # type: ignore
            mm_data = []
            filenames = []

            # Process images
            for image_resolution in req.data["image_resolution"]:  # type: ignore
                filename = create_dummy_image(
                    width=image_resolution[1], height=image_resolution[0], id=0
                )
                fn_tuple, mm_dict = create_mm_entry("image", filename)
                filenames.append(fn_tuple)
                mm_data.append(mm_dict)

            # Process videos
            for video_resolution in req.data["video_resolution"]:  # type: ignore
                filename = create_dummy_video(
                    num_frames=video_resolution[0],
                    width=video_resolution[2],
                    height=video_resolution[1],
                    id=0,
                )
                fn_tuple, mm_dict = create_mm_entry("video", filename)
                filenames.append(fn_tuple)
                mm_data.append(mm_dict)

            # Process audio
            for audio_duration in req.data["audio_duration"]:  # type: ignore
                filename = create_dummy_audio(duration_sec=int(audio_duration), id=0)
                fn_tuple, mm_dict = create_mm_entry("audio", filename)
                filenames.append(fn_tuple)
                mm_data.append(mm_dict)

            # Create request (VLM or Omni based on vlm_only flag)
            mm_enc = (
                sum(req.data["image_tokens"])  # type: ignore
                + sum(req.data["audio_tokens"])  # type: ignore
                + sum(req.data["video_tokens"])  # type: ignore
            )
            common_fields = {
                "prompt": prompt,
                "prompt_len": len(tokenizer(prompt).input_ids),
                "multi_modal_data_list": mm_data,
                "output_len": int(req.data["output_tokens"]),  # type: ignore
                "timestamp": req.timestamp,
                "filenames": filenames,
                "mm_encoder_tokens": mm_enc,
            }

            if self.vlm_only:
                sampled_req = VLMRequest(**common_fields)
            else:
                sampled_req = OmniRequest(
                    **common_fields,
                    return_audio=np.random.rand() < self.return_audio_prob,
                )

            sampled_requests.append(sampled_req)  # type: ignore

        if not self.vlm_only:
            print(
                f"Sampled {sum(1 for r in sampled_requests if r.return_audio)} requests with return_audio=True (prob={self.return_audio_prob})"  # type: ignore
            )

        save_requests(sampled_requests, path)

        return sampled_requests


# -----------------------------------------------------------------------------
# ReplayedServegen — replay from profiled servegen data
# -----------------------------------------------------------------------------

DATA_DIR = Path("data")


def _collect_servegen_pool(
    model_id: str,
    image_prob: float,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Scan profiled servegen result files and collect per-request characteristics.

    Returns the full list of (prompt_len, output_tokens, image_resolutions)
    from all matching profiling runs, preserving duplicates so that sampling
    from this pool reproduces the original request distribution (minus any
    requests that lack image_resolutions from pre-patch data).
    """
    pool: list[tuple[int, int, list[tuple[int, int]]]] = []
    num_files = 0
    skipped_no_res = 0

    for app_dir in ["llm", "monolithic-mllm"]:
        model_dir = DATA_DIR / app_dir / model_id
        if not model_dir.exists():
            continue
        for result_path in model_dir.rglob("result.json"):
            # Match servegen experiments with the right image_prob
            dir_name = result_path.parent.name
            if "servegen" not in dir_name:
                continue
            if f"img_prob{image_prob}" not in dir_name:
                continue

            try:
                with open(result_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Skip entire file if it's pre-patch (no image_resolutions on any result)
            file_results = data.get("results", [])
            if file_results and "image_resolutions" not in file_results[0]:
                skipped_no_res += len(file_results)
                continue

            num_files += 1
            for r in file_results:
                if not r.get("success", True):
                    continue
                img_res = r.get("image_resolutions")
                if img_res is None:
                    skipped_no_res += 1
                    continue
                prompt_len = r.get("prompt_len", 0)
                output_tokens = r.get("output_tokens", 0)
                img_res_list: list[tuple[int, int]] = [
                    (ir[0], ir[1]) for ir in img_res
                ]

                # Filter by image_prob: image requests must have images
                if image_prob >= 1.0 and len(img_res_list) == 0:
                    continue
                if image_prob <= 0.0 and len(img_res_list) > 0:
                    continue

                pool.append((prompt_len, output_tokens, img_res_list))

    unique_types = len({
        (pl, ot, tuple(tuple(ir) for ir in irs)) for pl, ot, irs in pool
    })
    msg = (
        f"ReplayedServegen: {len(pool)} replayable requests "
        f"({unique_types} unique types) from {num_files} profiling runs"
    )
    if skipped_no_res:
        msg += f" (skipped {skipped_no_res} pre-patch requests without image_resolutions)"
    logger.info(msg)
    print(msg)
    return pool


class ReplayedServegen(Dataset):
    """Replay servegen requests using only profiled (known) characteristics.

    Scans profiled servegen result files to collect per-request
    (prompt_len, output_tokens, image_resolutions) tuples, then samples
    from this known pool with Poisson timestamps.
    """

    request_rate: float
    duration: int
    image_prob: float
    random_seed: int = 0

    def to_suffix(self) -> str:
        suffix = (
            f"replayed_servegen+rate{self.request_rate}+duration{self.duration}"
            f"+img_prob{self.image_prob}"
        )
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    @with_saved_requests("VLM", VLMRequest)
    def sample(self, model_id: str) -> list[VLMRequest]:
        """Sample requests from profiled servegen data pool."""
        tokenizer = setup_sampling(model_id, self.random_seed)
        rng = np.random.RandomState(self.random_seed)

        # Collect known request characteristics from profiled data
        pool = _collect_servegen_pool(model_id, self.image_prob)
        if not pool:
            raise RuntimeError(
                f"No replayable servegen data found for {model_id} "
                f"with image_prob={self.image_prob}. "
                "Re-profile servegen workloads with the image_resolutions patch first."
            )

        num_requests = round(self.request_rate * self.duration)

        # Sample from pool (with replacement)
        indices = rng.randint(0, len(pool), size=num_requests)

        # Generate Poisson timestamps
        intervals = rng.exponential(1.0 / self.request_rate, size=num_requests)
        timestamps = np.cumsum(intervals).tolist()

        # Cache dummy images by resolution
        image_cache: dict[tuple[int, int], str] = {}

        sampled_requests: list[VLMRequest] = []
        for i, idx in enumerate(indices):
            prompt_len, output_tokens, image_resolutions = pool[idx]

            prompt = generate_random_text_with_length(tokenizer, prompt_len)
            mm_data = []
            filenames = []

            for w, h in image_resolutions:
                cache_key = (w, h)
                if cache_key not in image_cache:
                    image_cache[cache_key] = create_dummy_image(
                        width=w, height=h, id=0
                    )
                fn_tuple, mm_dict = create_mm_entry("image", image_cache[cache_key])
                filenames.append(fn_tuple)
                mm_data.append(mm_dict)

            sampled_requests.append(
                VLMRequest(
                    prompt=prompt,
                    prompt_len=len(tokenizer(prompt).input_ids),
                    multi_modal_data_list=mm_data,
                    output_len=output_tokens,
                    timestamp=timestamps[i],
                    filenames=filenames,
                )
            )

        return sampled_requests


# -----------------------------------------------------------------------------
# ReplayedOmniServegen — replay from profiled omni servegen data
# -----------------------------------------------------------------------------


from collections import namedtuple

OmniPoolEntry = namedtuple("OmniPoolEntry", [
    "prompt_len",          # text-only prompt tokens (excludes encoder tokens)
    "output_tokens",
    "image_resolutions",   # list[(h, w)]
    "video_resolutions",   # list[(frames, h, w)]
    "audio_durations",     # list[int] seconds
    "return_audio",        # bool
    "mm_encoder_tokens",   # total multimodal encoder tokens
])


def parse_omni_filenames(
    filenames: list[list[str]] | list[tuple[str, str]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[int]]:
    """Parse multimodal metadata from request filename tuples.

    Filename conventions (from mm_utils):
      - Image: ``{h}x{w}_{id}.png``
      - Video: ``{frames}x{h}x{w}_{id}.mp4``
      - Audio: ``{dur}s_{id}.wav``

    Returns:
        (image_resolutions, video_resolutions, audio_durations) where
        image_resolutions is list of (h, w), video_resolutions is list
        of (frames, h, w), and audio_durations is list of int seconds.
    """
    images: list[tuple[int, int]] = []
    videos: list[tuple[int, int, int]] = []
    audios: list[int] = []
    for modality, fn in filenames:
        stem = fn.rsplit("_", 1)[0]  # strip _{id}.ext
        if modality == "image":
            parts = stem.split("x")
            images.append((int(parts[0]), int(parts[1])))
        elif modality == "video":
            parts = stem.split("x")
            videos.append((int(parts[0]), int(parts[1]), int(parts[2])))
        elif modality == "audio":
            audios.append(int(stem.rstrip("s")))
    return images, videos, audios


PATCHED_DATA_DIR = Path("patched_data")


def collect_omni_servegen_pool(
    model_id: str,
    image_prob: float = 1.0,
    audio_prob: float = 0.5,
    video_prob: float = 0.5,
    steady_state_only: bool = True,
    patched_data_dir: Path | None = None,
) -> list[OmniPoolEntry]:
    """Collect per-request characteristics from patched profiling data.

    Reads result.json files from ``patched_data/`` which contain per-request
    multimodal metadata (image_resolutions, video_metadata, audio_durations_s,
    return_audio) directly — no cached requests.json needed.

    Falls back to the old method (reading from requests.json) if patched_data
    is not available.
    """
    data_dir = patched_data_dir or PATCHED_DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Patched data directory not found: {data_dir}. "
            "Run the data patching pipeline first."
        )

    # Scan all result.json files under patched_data for matching workload
    pool: list[OmniPoolEntry] = []
    num_files = 0
    num_failed = 0
    num_non_ss = 0

    model_short = model_id.split("/")[-1]
    for result_path in sorted(data_dir.rglob("result.json")):
        # Filter by model — check if model name appears in the path
        if model_short not in str(result_path):
            continue

        # Filter by workload probabilities — check the directory name
        dir_name = str(result_path.parent)
        if f"img_prob{image_prob}" not in dir_name:
            continue
        if f"audio_prob{audio_prob}" not in dir_name:
            continue
        if f"video_prob{video_prob}" not in dir_name:
            continue

        try:
            with open(result_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        results = data.get("results", [])
        if not results:
            continue

        # Check that this file has patched metadata (value must be non-None)
        sample = results[0]
        if sample.get("image_resolutions") is None:
            logger.debug(f"Skipping {result_path}: no image_resolutions (unpatched)")
            continue

        num_files += 1
        for r in results:
            if not r.get("success", True):
                num_failed += 1
                continue
            if steady_state_only and not r.get("steady_state", True):
                num_non_ss += 1
                continue

            # Skip entries missing patched metadata
            if r.get("image_resolutions") is None:
                continue

            # Extract multimodal metadata directly from result
            img_res = [tuple(res) for res in r.get("image_resolutions", [])]
            vid_meta = r.get("video_metadata", [])
            vid_res = [(v["frames"], v["height"], v["width"]) for v in vid_meta]
            aud_dur = r.get("audio_durations_s", [])
            return_audio = r.get("return_audio", False)

            mm_items = r.get("mm_items", [])
            mm_enc_tokens = sum(item.get("encoder_tokens", 0) for item in mm_items)

            prompt_len = r.get("prompt_len") or r.get("num_prompt_tokens", 0)
            if "num_prompt_tokens" in r and "prompt_len" not in r:
                prompt_len = max(0, prompt_len - mm_enc_tokens)
            output_tokens = r.get("output_tokens") or r.get("num_output_tokens", 0)

            # Skip entries with no output tokens (would cause num_output_seq=0)
            if output_tokens <= 0:
                continue

            pool.append(OmniPoolEntry(
                prompt_len=prompt_len,
                output_tokens=output_tokens,
                image_resolutions=img_res,
                video_resolutions=vid_res,
                audio_durations=aud_dur,
                return_audio=return_audio,
                mm_encoder_tokens=mm_enc_tokens,
            ))

    unique_signatures = len({
        (e.prompt_len, e.output_tokens,
         tuple(e.image_resolutions), tuple(e.video_resolutions),
         tuple(e.audio_durations), e.return_audio)
        for e in pool
    })
    msg = (
        f"ReplayedOmniServegen: {len(pool)} replayable requests "
        f"({unique_signatures} unique) from {num_files} patched result files"
    )
    if num_failed:
        msg += f", {num_failed} failed filtered"
    if num_non_ss:
        msg += f", {num_non_ss} non-SS filtered"
    logger.info(msg)
    print(msg)
    return pool


class ReplayedOmniServegen(Dataset):
    """Replay omni-modal requests with content from patched profiling data.

    Uses ServeGen timestamps by default (realistic burstiness). Set
    ``use_poisson=True`` for Poisson inter-arrivals instead.

    Pool is built from ``patched_data/`` result.json files which contain
    per-request multimodal metadata directly — no requests.json or mm_data
    directories needed for pool construction.
    """

    request_rate: float
    duration: int
    image_prob: float = 1.0
    audio_prob: float = 0.5
    video_prob: float = 0.5
    return_audio_prob: float = 0.2
    steady_state_only: bool = True
    use_poisson: bool = False
    random_seed: int = 0

    def to_suffix(self) -> str:
        suffix = (
            f"replayed_omni_servegen+rate{self.request_rate}"
            f"+duration{self.duration}"
            f"+img_prob{self.image_prob}+audio_prob{self.audio_prob}"
            f"+video_prob{self.video_prob}"
        )
        if self.return_audio_prob > 0.0:
            suffix += f"+return_audio_prob{self.return_audio_prob}"
        if self.steady_state_only:
            suffix += "+ss_only"
        if self.use_poisson:
            suffix += "+poisson"
        if self.random_seed != 0:
            suffix += f"+seed{self.random_seed}"
        return suffix

    @with_saved_requests("Omni", OmniRequest)
    def sample(self, model_id: str) -> list[OmniRequest]:
        """Sample requests: content from patched profiling pool, timestamps separate.

        Content (pool indices + return_audio flags) is determined solely by
        (random_seed, num_requests) so that e.g. 500s*4rps and 400s*5rps
        produce the identical 2000 request characteristics.  Timestamps are
        generated from a separate RNG and may vary with rate/duration.
        """
        tokenizer = setup_sampling(model_id, self.random_seed)

        pool = collect_omni_servegen_pool(
            model_id,
            image_prob=self.image_prob,
            audio_prob=self.audio_prob,
            video_prob=self.video_prob,
            steady_state_only=self.steady_state_only,
        )
        if not pool:
            raise RuntimeError(
                f"No replayable omni servegen data found for {model_id} "
                f"with img_prob={self.image_prob}, audio_prob={self.audio_prob}, "
                f"video_prob={self.video_prob}. "
                "Ensure patched_data/ directory has profiling results."
            )

        num_requests = round(self.request_rate * self.duration)

        # ── Content RNGs — each stream uses its own RNG so that the first
        # N requests of a larger run are identical to an N-request run
        # (prefix-stable across different num_requests values). ──
        index_rng = np.random.RandomState(self.random_seed)
        audio_rng = np.random.RandomState(self.random_seed + 2)
        indices = index_rng.randint(0, len(pool), size=num_requests)
        if self.return_audio_prob > 0.0:
            return_audio_flags = audio_rng.rand(num_requests) < self.return_audio_prob
        else:
            return_audio_flags = np.zeros(num_requests, dtype=bool)

        # ── Timestamps (separate RNG, does not affect content) ──
        if self.use_poisson:
            ts_rng = np.random.RandomState(self.random_seed + 1)
            intervals = ts_rng.exponential(
                1.0 / self.request_rate, size=num_requests,
            )
            timestamps = np.cumsum(intervals).tolist()
        else:
            sg_pool = ClientPool(
                Category.MULTIMODAL, "mm-image",
                data_dir="../third_party/servegen/data",
            )
            sg_view = sg_pool.span(0, float("inf"))
            rate_fn = get_scaled_rate_fn(sg_view, self.request_rate / 5.55)
            sg_requests = generate_workload(
                sg_view, rate_fn, duration=self.duration, seed=self.random_seed,
            )
            sg_timestamps = [r.timestamp for r in sg_requests]
            if len(sg_timestamps) >= num_requests:
                timestamps = sg_timestamps[:num_requests]
            else:
                # Pad with Poisson after last servegen timestamp
                ts_rng = np.random.RandomState(self.random_seed + 1)
                extra = num_requests - len(sg_timestamps)
                last_ts = sg_timestamps[-1] if sg_timestamps else 0.0
                extra_intervals = ts_rng.exponential(
                    1.0 / self.request_rate, size=extra,
                )
                timestamps = sg_timestamps + (
                    last_ts + np.cumsum(extra_intervals)
                ).tolist()

        # ── Build requests ──
        # Re-seed global RNG so that prompt text generation is deterministic
        # regardless of how much global state timestamp generation consumed.
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        image_cache: dict[tuple[int, int], str] = {}
        video_cache: dict[tuple[int, int, int], str] = {}
        audio_cache: dict[int, str] = {}

        sampled_requests: list[OmniRequest] = []
        num_skipped = 0
        for i, idx in enumerate(indices):
            entry = pool[idx]
            if entry.output_tokens <= 0 or entry.prompt_len <= 0:
                num_skipped += 1
                continue
            prompt = generate_random_text_with_length(tokenizer, entry.prompt_len)
            mm_data: list[dict[str, Any]] = []
            filenames: list[tuple[str, str]] = []

            for h, w in entry.image_resolutions:
                key = (h, w)
                if key not in image_cache:
                    image_cache[key] = create_dummy_image(height=h, width=w, id=0)
                fn_tuple, mm_dict = create_mm_entry("image", image_cache[key])
                filenames.append(fn_tuple)
                mm_data.append(mm_dict)

            for frames, h, w in entry.video_resolutions:
                key = (frames, h, w)
                if key not in video_cache:
                    video_cache[key] = create_dummy_video(
                        num_frames=frames, height=h, width=w, id=0,
                    )
                fn_tuple, mm_dict = create_mm_entry("video", video_cache[key])
                filenames.append(fn_tuple)
                mm_data.append(mm_dict)

            for dur in entry.audio_durations:
                if dur not in audio_cache:
                    audio_cache[dur] = create_dummy_audio(duration_sec=dur, id=0)
                fn_tuple, mm_dict = create_mm_entry("audio", audio_cache[dur])
                filenames.append(fn_tuple)
                mm_data.append(mm_dict)

            sampled_requests.append(
                OmniRequest(
                    prompt=prompt,
                    prompt_len=len(tokenizer(prompt).input_ids),
                    multi_modal_data_list=mm_data,
                    output_len=entry.output_tokens,
                    timestamp=timestamps[i],
                    filenames=filenames,
                    return_audio=bool(return_audio_flags[i]),
                    mm_encoder_tokens=entry.mm_encoder_tokens,
                )
            )

        if num_skipped:
            logger.warning(
                f"Skipped {num_skipped}/{num_requests} pool entries "
                f"with output_tokens<=0 or prompt_len<=0"
            )
            print(
                f"  WARNING: skipped {num_skipped}/{num_requests} invalid pool entries"
            )
        logger.info(
            f"Sampled {len(sampled_requests)} requests "
            f"({'Poisson' if self.use_poisson else 'ServeGen'} timestamps, "
            f"pool size={len(pool)})"
        )
        print(
            f"Sampled {len(sampled_requests)} requests with "
            f"{'Poisson' if self.use_poisson else 'ServeGen'} timestamps "
            f"(pool={len(pool)}, return_audio_prob={self.return_audio_prob})"
        )
        return sampled_requests


# -----------------------------------------------------------------------------
# Main / Testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    # Test ServeGenDataset (VLM mode)
    print("=" * 80)
    print("Testing ServeGenDataset (VLMRequest, vlm_only=True)")
    print("=" * 80)
    vlm_dataset = ServeGenDataset(
        request_rate=10,
        duration=60,
        image_prob=1.0,
        audio_prob=0,
        video_prob=0,
        vlm_only=True,
        random_seed=48105,
    )
    vlm_workload = vlm_dataset.sample(model_id=model_id)
    print(f"Sampled {len(vlm_workload)} VLM requests from ServeGen.")

    # Test ServeGenDataset (Omni mode)
    print("\n" + "=" * 80)
    print("Testing ServeGenDataset (OmniRequest, vlm_only=False)")
    print("=" * 80)
    omni_dataset = ServeGenDataset(
        request_rate=10,
        duration=60,
        image_prob=1.0,
        audio_prob=0.5,
        video_prob=0.5,
        return_audio_prob=0.3,
        vlm_only=False,
        random_seed=48105,
    )
    omni_workload = omni_dataset.sample(model_id=model_id)
    print(f"Sampled {len(omni_workload)} omni requests from ServeGen.")
    print(
        f"  - With return_audio=True: {sum(1 for r in omni_workload if r.return_audio)}"
    )  # type: ignore

    # Test ControlledImageChat
    print("\n" + "=" * 80)
    print("Testing ControlledImageChat")
    print("=" * 80)
    controlled_dataset = ControlledImageChat(
        num_requests=600,
        input_len=256,
        output_len=128,
        image_width=1920,
        image_height=1080,
        image_count=2,
        image_probability=0.8,
        random_seed=48105,
    )
    controlled_workload: list[VLMRequest] = controlled_dataset.sample(model_id=model_id)
    print(f"Sampled {len(controlled_workload)} controlled requests.")
    print(
        f"  - With images: {sum(1 for r in controlled_workload if len(r.multi_modal_data_list) > 0)}"
    )
    print(
        f"  - Without images: {sum(1 for r in controlled_workload if len(r.multi_modal_data_list) == 0)}"
    )
    print(
        f"  - Total images: {sum(len(r.multi_modal_data_list) for r in controlled_workload)}"
    )
    print(
        f"  - Avg images per request: {sum(len(r.multi_modal_data_list) for r in controlled_workload) / len(controlled_workload):.2f}"
    )

    # Test ControlledDiT
    print("\n" + "=" * 80)
    print("Testing ControlledDiT")
    print("=" * 80)
    dit_dataset = ControlledDiT(
        num_requests=600,
        num_inference_steps=50,
        resolution_pool=[(512, 512), (768, 768), (1024, 1024)],
        resolution_weights=[0.5, 0.3, 0.2],
        random_seed=48105,
    )
    dit_workload: list[DiTRequest] = dit_dataset.sample(model_id=model_id)
    print(f"Sampled {len(dit_workload)} DiT requests.")
    # Count resolutions
    from collections import Counter

    resolution_counts = Counter(
        (r.output_image_height, r.output_image_width) for r in dit_workload
    )
    print("  - Resolution distribution:")
    for res, count in sorted(resolution_counts.items()):
        print(
            f"    {res[0]}x{res[1]}: {count} ({count / len(dit_workload) * 100:.1f}%)"
        )
