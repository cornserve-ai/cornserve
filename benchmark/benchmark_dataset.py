# SPDX-License-Identifier: Apache-2.0
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - VisionArena
"""

import base64
import hashlib
import os
import io
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Optional, Union

from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

FILE_SERVER_URL = "http://ampere00.eecs.umich.edu:32000"

@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[dict] = None
    video_urls: list[str] = field(default_factory=list)
    audio_urls: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize the BenchmarkDataset with an optional dataset path and random
        seed.  Args:
            dataset_path (Optional[str]): Path to the dataset. If None, it
            indicates that a default or random dataset might be used.
            random_seed (int): Seed value for reproducible shuffling or
            sampling. Defaults to DEFAULT_SEED.
        """
        self.dataset_path = dataset_path
        # Set the random seed, ensuring that a None value is replaced with the
        # default seed.
        self.random_seed = (random_seed
                            if random_seed is not None else self.DEFAULT_SEED)
        self.data = None

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(self, tokenizer: PreTrainedTokenizerBase,
               num_requests: int) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

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

    def maybe_oversample_requests(self, requests: list[SampleRequest],
                                  num_requests: int) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests,
                                        k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.",
                        num_requests)


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
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len
                                                            < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


def save_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input, save it to local file, and return a multimedia content dictionary.

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
    
    if isinstance(image, dict) and 'bytes' in image:
        pil_image = Image.open(BytesIO(image['bytes']))
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
        raise ValueError(f"Invalid image input {image}. Must be a PIL.Image.Image"
                        " or str or dictionary with raw image bytes.")
    
    # Preserve original format if possible, otherwise default to PNG for lossless
    original_format = getattr(pil_image, 'format', None)
    
    # Determine format and extension
    if original_format in ['JPEG', 'JPG']:
        save_format = 'JPEG'
        extension = '.jpg'
        pil_image = pil_image.convert("RGB")  # JPEG doesn't support transparency
    elif original_format == 'PNG':
        save_format = 'PNG'
        extension = '.png'
        # Keep RGBA if it has transparency
    elif original_format in ['GIF', 'WEBP']:
        save_format = original_format
        extension = f'.{original_format.lower()}'
    else:
        # Default to PNG for unknown formats (lossless)
        save_format = 'PNG'
        extension = '.png'
    
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
            "image_url": {
                "url": f"{FILE_SERVER_URL}/images/{filename}"
            },
        }
    
    # Save image to local file
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    # Return structure with localhost URL
    return {
        "type": "image_url",
        "image_url": {
            "url": f"{FILE_SERVER_URL}/images/{filename}"
        },
    }

def process_image(image: Any) -> Mapping[str, Any]:
    # TODO: hash and save the image to a local file under videos, and return the http url
    """
    Process a single image input and return a multimedia content dictionary.

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
    if isinstance(image, dict) and 'bytes' in image:
        image = Image.open(BytesIO(image['bytes']))
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(
                image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            },
        }

    if isinstance(image, str):
        image_url = (image if image.startswith(
            ("http://", "file://")) else f"file://{image}")
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(f"Invalid image input {image}. Must be a PIL.Image.Image"
                     " or str or dictionary with raw image bytes.")


# -----------------------------------------------------------------------------
# HuggingFace Dataset Base Implementation
# -----------------------------------------------------------------------------
class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: Union[set[str], dict[str, Callable]] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        self.data = self.data.shuffle(seed=self.random_seed)


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(HuggingFaceDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {
        "lmarena-ai/VisionArena-Chat":
        lambda x: x["conversation"][0][0]["content"],
        "lmarena-ai/vision-arena-bench-v0.1":
        lambda x: x["turns"][0][0]["content"]
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
    ) -> list:
        output_len = (output_len
                      if output_len is not None else self.DEFAULT_OUTPUT_LEN)
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.dataset_path)
            if parser_fn is None:
                raise ValueError(
                    f"Unsupported dataset path: {self.dataset_path}")
            prompt = parser_fn(item)
            # mm_content = save_image(item["images"][0])
            mm_content = process_image(item["images"][0])
            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                ))
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests

