import sys
import random
from functools import cache

import numpy as np
import torch
import torch.distributed as dist
from transformers import BatchFeature
from torch.testing._internal.common_utils import FILE_SCHEMA
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase

from cornserve.task_executors.eric.schema import Modality
from cornserve.task_executors.eric.router.processor import ImageLoader, Processor
from cornserve.task_executors.eric.distributed.parallel import init_distributed, destroy_distributed


class TestModalityData:
    """Modality data for testing."""

    def __init__(self, url: str, modality: Modality) -> None:
        self.url = url
        self.modality = modality
        self.image = ImageLoader().load_from_url(url)

    @cache
    def processed(self, model_id: str) -> BatchFeature:
        processor = Processor(model_id, self.modality, 1)
        return processor._do_process(self.url)


class InferenceTestCase(MultiProcessTestCase):
    """Sets up and tears down a distributed environment for inference tests."""

    @property
    def world_size(self) -> int:
        raise NotImplementedError

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()
        destroy_distributed()

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe, **kwargs) -> None:
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        try:
            init_distributed(
                world_size=int(self.world_size),
                rank=self.rank,
                backend="nccl",
                init_method=f"{FILE_SCHEMA}{self.file_name}",
            )
            # dist.init_process_group(
            #     init_method=f"{FILE_SCHEMA}{self.file_name}",
            #     backend="nccl",
            #     world_size=int(self.world_size),
            #     rank=self.rank,
            # )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        # if torch.cuda.is_available() and (device_count := torch.cuda.device_count()):
        #     device_id = self.rank % torch.cuda.device_count()
        #     torch.cuda.set_device(device_id)

        self.reset_seed()

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier()

        # with (
        #     torch.backends.cudnn.flags(
        #         enabled=False, deterministic=True, benchmark=False
        #     ),
        #     patch.object(dist, "batch_isend_irecv", new=batch_isend_irecv_gloo),
        #     patch.object(dist, "all_to_all", new=all_to_all_gloo),
        #     patch.object(dist, "all_to_all_single", new=all_to_all_single_gloo),
        #     patch.object(dist, "reduce_scatter", new=reduce_scatter_gloo),
        #     patch(
        #         "colossalai.pipeline.p2p._check_device",
        #         return_value=(torch.device("cuda"), False),
        #     ),
        # ):
        #     torch.use_deterministic_algorithms(mode=True)
        #     self.run_test(test_name, parent_pipe)

        self.run_test(test_name, parent_pipe)

        try:
            dist.barrier()
        except RuntimeError:
            # Some processes may be hung due to synchronization error,
            # and return an error as other processes closed the connection.
            # Simply ignore the error here.
            pass

        # dist.destroy_process_group()

    def reset_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
