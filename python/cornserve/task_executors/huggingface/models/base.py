"""Base class for Hugging Face executor models."""

from abc import ABC, abstractmethod

from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse


class HFModel(ABC):
    """Base class for Hugging Face executor models."""

    @abstractmethod
    def generate(self, request: HuggingFaceRequest) -> HuggingFaceResponse:
        """Generate a response based on the request."""
