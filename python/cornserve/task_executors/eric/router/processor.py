"""Defines the Processor class for handling modality preprocessing."""

import time
import asyncio
import base64
import requests
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from transformers import AutoImageProcessor, BaseImageProcessor, BatchFeature

from cornserve.task_executors.eric.schema import Modality
from cornserve.logging import get_logger

logger = get_logger(__name__)


class Processor:
    """Runs modality processing on input data asynchronously in a thread pool."""

    def __init__(self, model_id: str, modality: Modality, num_workers: int) -> None:
        """Initialize the Processor."""
        self.model_id = model_id

        if modality == Modality.IMAGE:
            self.processor: BaseImageProcessor = AutoImageProcessor.from_pretrained(
                model_id
            )
            self.loader = ImageLoader()
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        self.loop = asyncio.get_event_loop()
        self.pool = ThreadPoolExecutor(max_workers=num_workers)

    def shutdown(self) -> None:
        """Shutdown the processor and the thread pool."""
        self.pool.shutdown(wait=True, cancel_futures=True)

    async def process(self, urls: list[str]) -> list[BatchFeature]:
        """Performs modality processing on input data in a thread pool."""
        # Here, we intentionally do not batch images in the processor because
        # 1. we are running them in parallel anyway, and
        # 2. the HF processor merges together processed tensors into a single tensor.
        return await asyncio.gather(
            *(
                self.loop.run_in_executor(self.pool, self._do_process, item)
                for item in urls
            )
        )

    def _do_process(self, url: str) -> BatchFeature:
        """Run processing on input data."""
        image = self.loader.load_from_url(url)
        return self.processor(image, return_tensors="np")


class ImageLoader:
    """Handles loading images from various sources, including URLs and base64 data."""

    def __init__(self) -> None:
        """Initialize the loader."""
        self.session = requests.Session()

    def load_from_url(self, url: str) -> Image.Image:
        """Load an image from a web, data, or file URL."""
        start_time = time.monotonic()

        url_spec = urlparse(url)
        if url_spec.scheme.startswith("http"):
            if url_spec.scheme not in ["http", "https"]:
                raise ValueError(f"Unsupported URL scheme: {url_spec.scheme}")
            with self.session.get(
                url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5
            ) as r:
                r.raise_for_status()
                image = self._load_bytes(r.content)
            logger.info(
                "Took %.3f seconds to load image from web URL",
                time.monotonic() - start_time,
            )
            return image

        elif url_spec.scheme == "data":
            data_spec, data = url_spec.path.split(",", 1)
            _, data_type = data_spec.split(";", 1)
            if data_type != "base64":
                raise ValueError(
                    f"Only base64 data URLs are supported; got '{data_type}'."
                )
            image = self._load_base64(data)
            logger.info(
                "Took %.3f seconds to load image from data URL",
                time.monotonic() - start_time,
            )
            return image

        elif url_spec.scheme == "file":
            image = self._load_file(url_spec.path)
            logger.info(
                "Took %.3f seconds to load image from file URL",
                time.monotonic() - start_time,
            )
            return image

        else:
            raise ValueError("The URL must be either a HTTP, data or file URL.")

    def _load_bytes(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data)).convert("RGB")

    def _load_base64(self, data: str) -> Image.Image:
        return self._load_bytes(base64.b64decode(data))

    def _load_file(self, filepath: str) -> Image.Image:
        return Image.open(filepath).convert("RGB")
