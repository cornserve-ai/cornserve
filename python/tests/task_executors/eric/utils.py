"""Testing utilities for Eric."""

import uuid
import subprocess
from functools import cache

import torch
import torch.nn as nn
import numpy.typing as npt

from cornserve.task_executors.eric.config import ImageDataConfig, ModalityConfig, VideoDataConfig
from cornserve.task_executors.eric.schema import Batch, Modality
from cornserve.task_executors.eric.router.processor import Processor


try:
    NUM_GPUS = int(
        subprocess
        .check_output(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits", "-i", "0"])
        .strip()
        .decode()
    )
except subprocess.CalledProcessError:
    NUM_GPUS = 0


class ModalityData:
    """Modality data for testing."""

    def __init__(self, url: str, modality: Modality) -> None:
        self.url = url
        self.modality = modality
        self.modality_config = ModalityConfig(
            num_workers=1,
            image_config=ImageDataConfig(),
            video_config=VideoDataConfig(max_num_frames=32),
        )

    @cache
    def processed(self, model_id: str) -> dict[str, npt.NDArray]:
        """Process the data for the given model."""
        processor = Processor(model_id, self.modality_config)
        return processor._do_process(self.modality, self.url)


def assert_same_weights(hf_model: nn.Module, our_model: nn.Module) -> None:
    """Ensure that parameters in the two models are the same."""
    hf_params = dict(hf_model.named_parameters())
    our_params = dict(our_model.named_parameters())
    assert len(hf_params) == len(our_params)
    for hf_name, hf_param in hf_params.items():
        our_param = our_params[hf_name]
        assert hf_param.shape == our_param.shape, hf_name
        assert torch.allclose(hf_param, our_param), hf_name


def batch_builder(model_id: str, images: list[ModalityData]) -> Batch:
    """Builds a Batch object to pass to ModelExecutor.execute_model."""
    data = {
        key: [torch.from_numpy(image.processed(model_id)[key]) for image in images]
        for key in images[0].processed(model_id).keys()
    }
    return Batch(
        modality=images[0].modality,
        request_ids=[uuid.uuid4().hex for _ in images],
        data_ids=[uuid.uuid4().hex for _ in images],
        data=data,
    )
