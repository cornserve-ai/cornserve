import torch

from cornserve.logging import get_logger

logger = get_logger(__name__)


class DeviceGroup:
    """A group of devices bound by a common process group.

    `torch.distributed` should be initialized before creating a device group.
    """

    def __init__(self, ranks: list[int], name: str) -> None:
        """Initialize the device group.

        Args:
            ranks: List of ranks in the group.
            name: Name of the group.
        """
        self.name = name
        self.ranks = ranks
        self.world_size = len(ranks)

        if self.world_size == 1:
            self.process_group = None
            self.rank = 0
            return

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                f"Distributed process group is not initialized. Cannot create device group {name}."
            )

        self.process_group = torch.distributed.new_group(ranks=ranks)
        self.rank = torch.distributed.get_rank(self.process_group)

        logger.info(
            "Device group %s initialized with ranks %s and world size %d.",
            name,
            ranks,
            self.world_size,
        )

    def shutdown(self) -> None:
        """Shutdown the device group."""
        if self.process_group is not None:
            torch.distributed.destroy_process_group(self.process_group)
            logger.info("Device group %s destroyed.", self.name)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Perform AllGather on the tensor across the device group.

        Args:
            input_: Tensor to AllGather.
            dim: Dimension to gather along.

        Returns:
            Gathered tensor.
        """
        world_size = self.world_size
        if world_size == 1:
            return input_

        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * world_size,) + input_size[1:]
        # Allocate output tensor.
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather.
        torch.distributed.all_gather_into_tensor(
            output_tensor, input_, group=self.process_group
        )
        # Reshape
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :],
        )
        return output_tensor

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Perform AllReduce on the tensor across the device group.

        AllReduce is performed in-place, so the input tensor is modified.

        Args:
            input_: Tensor to AllReduce.

        Returns:
            Reduced tensor.
        """
        world_size = self.world_size
        if world_size == 1:
            return input_

        # In-place AllReduce.
        torch.distributed.all_reduce(input_, group=self.process_group)
        return input_


TP_GROUP: DeviceGroup | None = None


def get_tensor_parallel_group() -> DeviceGroup:
    """Get the global tensor parallel group.

    This is expected to work even when we're not doing distributed inference.
    Collective calls will be no-ops, world size will be 1, and rank will be 0.
    """
    if TP_GROUP is None:
        raise RuntimeError("Tensor parallel group is not initialized.")
    return TP_GROUP


def init_distributed(
    world_size: int,
    rank: int,
    backend: str = "nccl",
    init_method: str = "tcp://127.0.0.1:29500",
) -> None:
    """Initialize the distributed process group."""
    if torch.distributed.is_initialized():
        logger.warning(
            "Distributed process group is already initialized. Skipping initialization."
        )
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    else:
        logger.warning(
            "CUDA is not available. Continuing to initialize distributed environment without CUDA."
        )

    # Only initialize if world size is greater than 1
    if world_size > 1:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        logger.info(
            "Distributed process group initialized with world size %d and rank %d.",
            world_size,
            rank,
        )

    # Initialize global tensor parallel group.
    # If world size is 1, it will not create a new group.
    global TP_GROUP
    TP_GROUP = DeviceGroup(
        ranks=list(range(world_size)),
        name="tensor_parallel_group",
    )


def destroy_distributed() -> None:
    """Destroy the distributed process group."""
    if not torch.distributed.is_initialized():
        logger.warning(
            "Distributed process group is not initialized. Skipping destruction."
        )
        return

    get_tensor_parallel_group().shutdown()
    torch.distributed.destroy_process_group()
    logger.info("Uninitialized distributed process groups.")
