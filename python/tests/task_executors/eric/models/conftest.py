import pytest
from PIL import Image
import torch

from cornserve.task_executors.eric.distributed.parallel import init_distributed
from cornserve.task_executors.eric.router.processor import ImageLoader


@pytest.fixture
def large_image_url() -> str:
    """Test image sized (3722, 2353)"""
    return "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"


@pytest.fixture
def large_image_pil(test_image_url: str) -> Image.Image:
    return ImageLoader().load_from_url(test_image_url)


@pytest.fixture
def init_gpu_inference_env():
    """Fixture to set up a GPU inference environment."""
    def inner(num_gpus: int):
        torch.set_grad_enabled(False)

        init_distributed(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            world_size=1,
            rank=0,
        )

    return inner
