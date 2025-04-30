"""Defines the Processor class for handling modality preprocessing."""

from __future__ import annotations

import asyncio
import base64
import importlib
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.parse import urlparse

import cv2
import cv2.videoio_registry as vr
import numpy as np
import numpy.typing as npt
import requests
from opentelemetry import trace
from PIL import Image
from transformers.models.auto.configuration_auto import AutoConfig

from cornserve.logging import get_logger
from cornserve.task_executors.eric.api import EmbeddingData
from cornserve.task_executors.eric.config import (
    ModalityConfig,
)
from cornserve.task_executors.eric.models.base import BaseModalityProcessor
from cornserve.task_executors.eric.models.registry import MODEL_REGISTRY
from cornserve.task_executors.eric.schema import Modality, ProcessedEmbeddingData

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
thread_local = threading.local()


class Processor:
    """Runs modality processing on input data asynchronously in a thread pool.

    Holds an instance of BaseModalityProcessor for the given model_id and modality,
    which invokes the appropriate processor for the given modality.
    """

    def __init__(self, model_id: str, modality_config: ModalityConfig) -> None:
        """Initialize the Processor."""
        # Saved variables
        self.model_id = model_id

        # Load the model config from HF and Eric model class
        hf_config = AutoConfig.from_pretrained(model_id)
        try:
            registry_entry = MODEL_REGISTRY[hf_config.model_type]
        except KeyError as e:
            raise ValueError(
                f"Model {model_id} (model_type={hf_config.model_type}) not found in model registry."
            ) from e
        model_module = importlib.import_module(f"cornserve.task_executors.eric.models.{registry_entry.module}")

        # Instantiate the thread pool
        def init_thread() -> None:
            """Initialize the thread-local processor and loader."""
            thread_local.loader = ModalityDataLoader(modality_config)
            thread_local.processor = model_module.ModalityProcessor(model_id)

        # Initialize for the main thread as well, which is useful for testing
        init_thread()

        self.loop = asyncio.get_event_loop()
        self.pool = ThreadPoolExecutor(
            max_workers=modality_config.num_workers,
            initializer=init_thread,
        )

    def shutdown(self) -> None:
        """Shutdown the processor and the thread pool."""
        self.pool.shutdown(wait=True, cancel_futures=True)

    @tracer.start_as_current_span(name="Processor.process")
    async def process(self, data: list[EmbeddingData]) -> list[ProcessedEmbeddingData]:
        """Performs modality processing on input data in a thread pool."""
        # Here, we intentionally do not batch images in the processor because
        # 1. we are running them in parallel anyway, and
        # 2. the HF processor merges together processed tensors into a single tensor.
        features = await asyncio.gather(
            *(self.loop.run_in_executor(self.pool, self._do_process, item.modality, item.url) for item in data)
        )
        processed = [
            ProcessedEmbeddingData(
                id=item.id,
                modality=item.modality,
                data=feature,
                receiver_sidecar_ranks=item.receiver_sidecar_ranks,
            )
            for item, feature in zip(data, features, strict=True)
        ]
        self._check_processed_data(processed)
        span = trace.get_current_span()
        for data_item in processed:
            for k, v in data_item.data.items():
                span.set_attribute(
                    f"processor.processed_data.{data_item.id}.{k}.shape",
                    v.shape,
                )
        return processed

    @tracer.start_as_current_span(name="Processor._do_process")
    def _do_process(self, modality: Modality, url: str) -> dict[str, npt.NDArray]:
        """Run processing on input data."""
        loader: ModalityDataLoader = thread_local.loader
        processor: BaseModalityProcessor = thread_local.processor
        data = loader.load_from_url(modality, url)
        with tracer.start_as_current_span(name="ModalityProcessor.process"):
            return processor.process(modality, data)

    def _check_processed_data(self, processed: list[ProcessedEmbeddingData]) -> None:
        """Check that all processed data is valid."""
        for item in processed:
            if not isinstance(item.data, dict):
                raise ValueError(f"Processed data should be a dict; got {type(item.data)}.")
            if not all(isinstance(v, np.ndarray) for v in item.data.values()):
                raise ValueError(f"All processed data should be numpy arrays; got {item.data}")


class BaseLoader(ABC):
    """Base class for downloading data from various sources."""

    @abstractmethod
    def load_bytes(self, data: bytes) -> npt.NDArray:
        """Load data from bytes."""

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        """Load data from base64 string."""

    @abstractmethod
    def load_file(self, filepath: str) -> npt.NDArray:
        """Load data from file path."""


class ImageLoader(BaseLoader):
    """Handles loading images from various sources."""

    def __init__(self, config: ModalityConfig) -> None:
        """Initialize the loader."""
        if config.image_config is None:
            raise ValueError("Image config must be set.")

        self.config = config

    def load_bytes(self, data: bytes) -> npt.NDArray:
        """Load image data from bytes."""
        return np.asarray(Image.open(BytesIO(data)).convert("RGB"))

    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        """Load image data from base64 string."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: str) -> npt.NDArray:
        """Load image data from file path."""
        return np.asarray(Image.open(filepath).convert("RGB"))


class VideoLoader(BaseLoader):
    """Handles loading videos from various sources."""

    def __init__(self, config: ModalityConfig) -> None:
        """Initialize the loader."""
        if config.video_config is None:
            raise ValueError("Video config must be set.")

        self.config = config
        self.max_num_frames = config.video_config.max_num_frames

        self.image_loader = ImageLoader(config)

    def load_bytes(self, data: bytes) -> npt.NDArray:
        """Load video data from bytes."""
        api_preference = None
        for candidate in vr.getStreamBufferedBackends():
            if not vr.hasBackend(candidate):
                continue
            if not vr.isBackendBuiltIn(candidate):
                _, abi, api = vr.getStreamBufferedBackendPluginVersion(candidate)
                if abi < 1 or (abi == 1 and api < 2):
                    continue
            api_preference = candidate
            break

        cap = cv2.VideoCapture(BytesIO(data), api_preference, [])  # type: ignore
        if not cap.isOpened():
            raise ValueError("Failed to open video stream.")

        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.max_num_frames is not None and total_num_frames > self.max_num_frames:
            frame_indices = np.linspace(0, total_num_frames - 1, self.max_num_frames, dtype=int)
        else:
            frame_indices = np.arange(total_num_frames, dtype=int)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((len(frame_indices), height, width, 3), dtype=np.uint8)

        frames_loaded = 0
        for i in range(total_num_frames):
            ret = cap.grab()
            if not ret:
                break
            if i in frame_indices:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                frames[frames_loaded] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_loaded += 1
        cap.release()

        assert frames_loaded == len(frame_indices), f"Expected {len(frame_indices)} frames, but got {frames_loaded}."

        return frames

    def load_base64(self, media_type: str, data: str) -> npt.NDArray:
        """Load video data from base64 string."""
        # Video as a sequence of JPEG frames
        if media_type.lower() == "video/jpeg":
            return np.stack([self.image_loader.load_base64("image/jpeg", frame) for frame in data.split(",")])

        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: str) -> npt.NDArray:
        """Load video data from file path."""
        with open(filepath, "rb") as f:
            data = f.read()

        return self.load_bytes(data)


class ModalityDataLoader:
    """Handles loading data for different modalities.

    This class is responsible for loading data from URLs using the appropriate
    modality loader.
    """

    def __init__(self, config: ModalityConfig) -> None:
        """Initialize the data loader."""
        self.session = requests.Session()

        self.image_loader = ImageLoader(config)
        self.video_loader = VideoLoader(config)

    @tracer.start_as_current_span(name="ModalityDataLoader.load_from_url")
    def load_from_url(self, modality: Modality, url: str) -> npt.NDArray:
        """Load an image from a web, data, or file URL."""
        match modality:
            case Modality.IMAGE:
                loader = self.image_loader
            case Modality.VIDEO:
                loader = self.video_loader
            case _:
                raise ValueError(f"Unsupported modality: {modality}")

        start_time = time.monotonic()

        span = trace.get_current_span()
        span.set_attribute("data.url", url)
        span.set_attribute("data.modality", modality.value)

        url_spec = urlparse(url)
        if url_spec.scheme.startswith("http"):
            if url_spec.scheme not in ["http", "https"]:
                raise ValueError(f"Unsupported URL scheme: {url_spec.scheme}")
            with self.session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5) as r:
                r.raise_for_status()
                data = loader.load_bytes(r.content)
            logger.info(
                "Took %.3f seconds to load %s from web URL",
                time.monotonic() - start_time,
                modality.value,
            )

        elif url_spec.scheme == "data":
            data_spec, data = url_spec.path.split(",", 1)
            media_type, data_type = data_spec.split(";", 1)
            if data_type != "base64":
                raise ValueError(f"Only base64 data URLs are supported; got '{data_type}'.")
            data = loader.load_base64(media_type, data)
            logger.info(
                "Took %.3f seconds to load %s from data URL",
                time.monotonic() - start_time,
                modality.value,
            )

        elif url_spec.scheme == "file":
            data = loader.load_file(url_spec.path)
            logger.info(
                "Took %.3f seconds to load %s from file URL",
                time.monotonic() - start_time,
                modality.value,
            )

        else:
            raise ValueError("The URL must be either a HTTP, data or file URL.")

        span.set_attribute("data.shape", data.shape)

        return data
