"""Testing utilities for Eric."""

import os
import subprocess
import uuid
from fnmatch import fnmatch
from functools import cache
from typing import Callable

import numpy.typing as npt
import torch
import torch.nn as nn

from cornserve.task_executors.eric.config import ImageDataConfig, ModalityConfig, VideoDataConfig
from cornserve.task_executors.eric.router.processor import Processor
from cornserve.task_executors.eric.schema import Modality, WorkerBatch

DUMP_DIR = os.getenv("CORNSERVE_TEST_DUMP_TENSOR_DIR", None)

try:
    NUM_GPUS = int(
        subprocess.check_output(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits", "-i", "0"])
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


def assert_same_weights(
    hf_model: nn.Module,
    our_model: nn.Module,
    required_prefixes: list[str] = [],
    ignored_prefixes: list[str] = [],
    transformed_weights: dict[str, Callable] = {},
) -> None:
    """Ensure that parameters in the two models are the same.

    Args:
        hf_model: Hugging Face model.
        our_model: Our model.
        required_prefixes: If given, only check parameters with these prefixes.
        ignored_prefixes: If given, ignore parameters with these prefixes.
        transformed_patterns: A mapping of patterns to functions that will be
            called to check weight equivalence. The function should take three args:
            our_name (str), our_param (Tensor), and hf_params (dict[str, Tensor]).
    """

    def filter_weight_dict(weight_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Filter the weight dictionary based on prefixes."""
        for k in list(weight_dict.keys()):
            if not any(k.startswith(prefix) for prefix in required_prefixes):
                del weight_dict[k]
                continue
            if any(k.startswith(prefix) for prefix in ignored_prefixes):
                del weight_dict[k]
        return weight_dict

    def check_param(name: str, param: torch.Tensor, hf_params: dict[str, torch.Tensor]) -> None:
        """Check if the parameter is the same, while accounting for transformed weights."""
        hf_param = hf_params.get(name)
        if hf_param is None:
            for pattern, func in transformed_weights.items():
                if fnmatch(name, pattern):
                    func(name, param, hf_params)
                    return
            raise ValueError(f"Parameter {name} not found in Hugging Face model")
        assert param.shape == hf_param.shape, name
        assert torch.allclose(param, hf_param), name

    hf_params = filter_weight_dict(dict(hf_model.named_parameters()))
    our_params = filter_weight_dict(dict(our_model.named_parameters()))
    if not transformed_weights:
        assert len(hf_params) == len(our_params)

    for name, param in our_params.items():
        check_param(name, param, hf_params)


def batch_builder(model_id: str, nickname: str, data: list[ModalityData]) -> WorkerBatch:
    """Builds a Batch object to pass to ModelExecutor.execute_model."""
    modality = data[0].modality
    assert all(item.modality == modality for item in data)

    processed_data = {
        key: [torch.from_numpy(item.processed(model_id)[key]) for item in data]
        for key in data[0].processed(model_id).keys()
    }
    batch = WorkerBatch(
        modality=data[0].modality,
        request_ids=[uuid.uuid4().hex for _ in data],
        data_ids=[uuid.uuid4().hex for _ in data],
        chunk_ids=[0 for _ in data],
        num_chunks=[1 for _ in data],
        receiver_ranks=[None for _ in data],
        data=processed_data,
        otel_carriers=[None for _ in data],
    )

    if DUMP_DIR is not None:
        batch._dump_prefix = f"{DUMP_DIR}/{nickname}-{modality.value}"

    return batch
