# SPDX-License-Identifier: Apache-2.0
"""This module defines a framework for sampling benchmark requests from various datasets.

Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - VisionArena

This file is based on https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_dataset.py
"""

import base64
import hashlib
import io
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field
import numpy as np
from io import BytesIO

from datasets import load_dataset
from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from servegen import Category, ClientPool
from servegen.construct import generate_workload
from servegen.utils import get_scaled_rate_fn, get_constant_rate_fn
from servegen_distributions import add_update_multimodal_content

from image_utils import create_dummy_image, create_dummy_video, create_dummy_audio, get_image_data_uris

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

FILE_SERVER_URL = "http://localhost:32000"


@dataclass
class SampleRequest:
    """Represents a single inference request for benchmarking."""

    prompt: str | Any
    prompt_len: int
    expected_output_len: int
    # original image
    multi_modal_data_list: list[dict[str, Any]]

    # synthetic data, will be procesed in transform()
    image_urls: list[str] = field(default_factory=list)
    # TODO: homogeneous synthetic audio & video
    # audio_urls: list[str] = field(default_factory=list)
    # video_urls: list[str] = field(default_factory=list)

    # mm data bookeeping
    filenames: list[str] = field(default_factory=list)

    # servegen
    timestamp: float | None = None

    # additional args
    encoder_fission: bool = False


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    """Base class for benchmark datasets."""

    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: str | None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """Initialize the BenchmarkDataset with an optional dataset path and random seed.

        Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.data = None

    def load_data(self) -> None:
        """Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError("load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase, num_requests: int) -> list[SampleRequest]:
        """Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(self, requests: list[SampleRequest], num_requests: int) -> None:
        """Oversamples the list of requests if its size is less than the desired number.

        Args:
            requests (List[SampleRequest]): The current list of sampled requests.
            num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests, k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.", num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long or combined_too_long)


def save_image(image: Any) -> dict[str, Any]:
    """Process a single image input, save it to local file, and return a multimedia content dictionary.

    Supports three input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG to local file.  - Uses content hash as filename.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Loads the image from the path/URL and saves locally.

    The image is saved to ./videos/ directory with filename based on content hash.
    Returns a dictionary with the localhost URL pointing to the saved file.

    Raises:
        ValueError: If the input is not a supported type.
        OSError: If there are issues creating directory or saving file.
    """
    # Ensure videos directory exists
    os.makedirs("./videos", exist_ok=True)

    # Process different input types to get PIL Image
    pil_image = None

    if isinstance(image, dict) and "bytes" in image:
        pil_image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        # Handle URL or file path
        if image.startswith(("http://", "https://")):
            # For URLs, you'd need to download first (requires requests library)
            # This is a simplified version - you may want to add proper URL handling
            raise NotImplementedError("URL downloading not implemented in this example")
        else:
            # Local file path
            file_path = image if image.startswith("file://") else image
            if file_path.startswith("file://"):
                file_path = file_path[7:]  # Remove file:// prefix
            pil_image = Image.open(file_path)
    else:
        raise ValueError(
            f"Invalid image input {image}. Must be a PIL.Image.Image or str or dictionary with raw image bytes."
        )

    # Preserve original format if possible, otherwise default to PNG for lossless
    original_format = getattr(pil_image, "format", None)

    # Determine format and extension
    if original_format in ["JPEG", "JPG"]:
        save_format = "JPEG"
        extension = ".jpg"
        pil_image = pil_image.convert("RGB")  # JPEG doesn't support transparency
    elif original_format == "PNG":
        save_format = "PNG"
        extension = ".png"
        # Keep RGBA if it has transparency
    elif original_format in ["GIF", "WEBP"]:
        save_format = original_format
        extension = f".{original_format.lower()}"
    else:
        # Default to PNG for unknown formats (lossless)
        save_format = "PNG"
        extension = ".png"

    # Save in memory to get bytes for hashing
    with io.BytesIO() as temp_buffer:
        pil_image.save(temp_buffer, format=save_format)
        image_bytes = temp_buffer.getvalue()

    # Generate hash from image content
    content_hash = hashlib.sha256(image_bytes).hexdigest()
    filename = f"{content_hash}{extension}"
    filepath = os.path.join("./images", filename)
    if os.path.exists(filepath):
        return {
            "type": "image_url",
            "image_url": {"url": f"{FILE_SERVER_URL}/images/{filename}"},
        }

    # Save image to local file
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    # Return structure with localhost URL
    return {
        "type": "image_url",
        "image_url": {"url": f"{FILE_SERVER_URL}/images/{filename}"},
    }


def process_image(image: Any) -> dict[str, Any]:
    """Process a single image input and return a multimedia content dictionary.

    Supports three input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG in memory.  - Encodes the JPEG data as a base64 string.  - Returns
       a dictionary with the image as a base64 data URL.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }

    if isinstance(image, str):
        image_url = image if image.startswith(("http://", "file://")) else f"file://{image}"
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image}. Must be a PIL.Image.Image or str or dictionary with raw image bytes."
    )


# -----------------------------------------------------------------------------
# HuggingFace Dataset Base Implementation
# -----------------------------------------------------------------------------
class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: set[str] | dict[str, Callable] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        dataset_subset: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the HuggingFaceDataset with dataset path, split, and subset."""
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,  # type: ignore
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        self.data = self.data.shuffle(seed=self.random_seed)


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


# helper function
def _enforce_input_len(
    tokenizer,
    text: str,
    target_len: int,
) -> str:
    """Enforce the input text to have a specific length in tokens."""
    # Use ONLY tokens from this text to pad up to target_len (vLLM counting).
    base_ids = tokenizer(text, add_special_tokens=False).input_ids
    # specials overhead for this prompt
    with_special = tokenizer(text, add_special_tokens=True).input_ids
    overhead = len(with_special) - len(base_ids)
    target_base = max(0, target_len - overhead)

    if len(base_ids) == 0:
        # Degenerate case: empty prompt; nothing to reuse.
        # Tiny fallback so we can build something to cycle.
        base_ids = tokenizer(" ", add_special_tokens=False).input_ids

    # If short, pad by cycling its own tokens
    if len(base_ids) < target_base:
        orig = base_ids[:]  # remember original tokens for cycling
        need = target_base - len(base_ids)
        while need > 0:
            take = min(len(orig), need)
            base_ids.extend(orig[:take])
            need -= take

    # If long, trim to target_base
    elif len(base_ids) > target_base:
        trunc_side = getattr(tokenizer, "truncation_side", "right")
        base_ids = base_ids[:target_base] if trunc_side == "right" else base_ids[-target_base:]

    # Decode once, then correct any decode→encode drift
    text = tokenizer.decode(base_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    # Small correction loop to land exactly on target_len with add_special_tokens=True
    for _ in range(4):
        cur = len(tokenizer(text, add_special_tokens=True).input_ids)
        if cur == target_len:
            break
        if cur < target_len:
            # top up by cycling original request tokens again
            more_needed = target_len - cur
            orig = tokenizer(text, add_special_tokens=False).input_ids  # current base after previous steps
            if not orig:
                # should not happen, but guard
                orig = tokenizer(" ", add_special_tokens=False).input_ids
            add_ids = []
            while more_needed > 0:
                take = min(len(orig), more_needed)
                add_ids.extend(orig[:take])
                more_needed -= take
            # append and re-decode
            base_ids = orig + add_ids
            text = tokenizer.decode(base_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        else:
            # trim the excess from base tokens
            over = cur - target_len
            base_ids = tokenizer(text, add_special_tokens=False).input_ids
            trunc_side = getattr(tokenizer, "truncation_side", "right")
            base_ids = base_ids[: max(0, len(base_ids) - over)] if trunc_side == "right" else base_ids[over:]
            text = tokenizer.decode(base_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    return text


class VisionArenaDataset(HuggingFaceDataset):
    """Vision Arena Dataset."""

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {
        "lmarena-ai/VisionArena-Chat": lambda x: x["conversation"][0][0]["content"],
        "lmarena-ai/vision-arena-bench-v0.1": lambda x: x["turns"][0][0]["content"],
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: int | None = None,
        input_len: int | None = None,
    ) -> list:
        """Sample requests from the Vision Arena dataset."""
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.dataset_path)  # type: ignore
            if parser_fn is None:
                raise ValueError(f"Unsupported dataset path: {self.dataset_path}")
            prompt = parser_fn(item)
            mm_content = process_image(item["images"][0])  # type: ignore

            # measure tokens without specials
            prompt_input_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_input_ids)

            if input_len is not None:
                prompt = _enforce_input_len(tokenizer, prompt, input_len)
                prompt_len = len(tokenizer(prompt).input_ids)

            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data_list=[mm_content],
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests

# https://github.com/ml-energy/leaderboard-v3/blob/master/mlenergy/llm/datasets.py
def _generate_random_text_with_length(
        tokenizer: PreTrainedTokenizerBase, target_length: int
   ) -> str:
       """Generate random text that tokenizes to approximately the target length.

       Args:
           tokenizer: Tokenizer to use for measuring token length.
           target_length: Target number of tokens.

       Returns:
           Random text string that tokenizes to approximately target_length tokens.
       """
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
       prompt = tokenizer.decode(
           current_tokens, clean_up_tokenization_spaces=True
       ).strip()

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

       prompt = tokenizer.decode(
           current_tokens, clean_up_tokenization_spaces=True
       ).strip()
       return prompt

class ServeGenDataset:

    DEFAULT_SEED = 0

    def __init__(
        self,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        self.random_seed = random_seed

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        request_rate: float,
        duration: int,
        no_image_prob: float,
        audio_prob: float,
        video_prob: float,
    ) -> list[SampleRequest]:

        np.random.seed(self.random_seed)
        print("\n=== Generating Multimodal Workload ===")

        # image
        pool = ClientPool(Category.MULTIMODAL, "mm-image", data_dir="../third_party/servegen/data")
        print(f"Loaded {len(pool.clients)} multimodal clients")
        view = pool.span(0, float("inf"))

        # rate_fn = get_constant_rate_fn(view, request_rate)
        rate_fn = get_scaled_rate_fn(view, request_rate / 5.55)

        requests = generate_workload(view, rate_fn, duration=duration, seed=self.random_seed)

        # Add video and audio data sampled from empirically fit distributions
        # based on video_prob and audio_prob. Remove images based on no_image_prob.
        requests = add_update_multimodal_content(
            requests,
            no_image_prob=no_image_prob,
            video_prob=video_prob,
            audio_prob=audio_prob,
            seed=self.random_seed,
        )

        # Print statistics
        print("\nMultimodal Workload Statistics:")
        print(f"Total Requests: {len(requests)}")
        print(f"Time Range: {requests[0].timestamp:.2f} to {requests[-1].timestamp:.2f}")
        
        # Calculate average token counts
        avg_text = sum(r.data['text_tokens'] for r in requests) / len(requests)  # type: ignore
        avg_output = sum(r.data['output_tokens'] for r in requests) / len(requests)  # type: ignore
        avg_image = sum(len(r.data['image_tokens']) for r in requests) / len(requests)  # type: ignore
        avg_audio = sum(len(r.data['audio_tokens']) for r in requests) / len(requests)  # type: ignore
        avg_video = sum(len(r.data['video_tokens']) for r in requests) / len(requests)  # type: ignore
        
        print("\nAverage Token Counts:")
        print(f"  Text tokens: {avg_text:.2f}")
        print(f"  Output tokens: {avg_output:.2f}")
        print(f"  Image count: {avg_image:.2f}")
        print(f"  Audio count: {avg_audio:.2f}")
        print(f"  Video count: {avg_video:.2f}")

        sampled_requests: list[SampleRequest] = []
        for req in requests:
            input_len = req.data["text_tokens"]
            prompt = _generate_random_text_with_length(tokenizer, input_len)  # type: ignore
            mm_data = []
            filenames = []

            # "image_resolution" will be an empty list if there are no images
            for image_resolution in req.data["image_resolution"]:  # type: ignore
                # for simplicity, we always use the same image for each resolution
                image_filename = create_dummy_image(
                    width=image_resolution[1],
                    height=image_resolution[0],
                    id=0,
                )
                # base64 data uri
                filenames.append(image_filename)
                image_uri = get_image_data_uris([image_filename], root="images")[0]
                mm_data.append({"type": "image_url", "image_url": {"url": image_uri}})

            for video_resolution in req.data["video_resolution"]:  # type: ignore
                video_filename = create_dummy_video(
                    num_frames=video_resolution[0],
                    width=video_resolution[2],
                    height=video_resolution[1],
                    id=0,
                )
                filenames.append(video_filename)
                video_uri = get_image_data_uris([video_filename], root="videos")[0]
                mm_data.append({"type": "video_url", "video_url": {"url": video_uri}})

            for audio_duration in req.data["audio_duration"]:  # type: ignore
                audio_filename = create_dummy_audio(duration_sec=int(audio_duration), id=0)
                filenames.append(audio_filename)
                audio_uri = get_image_data_uris([audio_filename], root="audios")[0]
                mm_data.append({"type": "audio_url", "audio_url": {"url": audio_uri}})

            sampled_req = SampleRequest(
                prompt=prompt,
                prompt_len=len(tokenizer(prompt).input_ids),
                multi_modal_data_list=mm_data,
                expected_output_len=req.data["output_tokens"],  # type: ignore
                timestamp=req.timestamp,
            )
            sampled_requests.append(sampled_req)

        return sampled_requests


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )
    sampled_requests: list[SampleRequest] = VisionArenaDataset(
        dataset_path="lmarena-ai/VisionArena-Chat",
        dataset_subset=None,
        dataset_split="train",
        random_seed=48105,
    ).sample(
        num_requests=2000,
        tokenizer=tokenizer,
        output_len=300,
        input_len=500,
    )

    # calculate input len mean and std
    print(f"Sampled {len(sampled_requests)} requests.")
    input_lens = []
    for req in sampled_requests:
        input_ids = tokenizer(req.prompt)["input_ids"]
        input_lens.append(len(input_ids))
    input_len_mean = sum(input_lens) / len(input_lens)
    input_len_std = (sum((x - input_len_mean) ** 2 for x in input_lens) / len(input_lens)) ** 0.5
    print(f"Input length mean: {input_len_mean}, std: {input_len_std}")
