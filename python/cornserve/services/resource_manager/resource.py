"""The Resource class represents all cluster resources."""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Literal

from cornserve.logging import get_logger

logger = get_logger(__name__)


class GPU:
    """GPU resource."""

    def __init__(self, node: str, global_rank: int, local_rank: int) -> None:
        """Initialize the GPU resource.

        Args:
            node: Node name
            global_rank: Cluster-wide unique global rank of the GPU
            local_rank: Local rank of the GPU on the node
        """
        self.node = node
        self.global_rank = global_rank
        self.local_rank = local_rank

        self.owner: str | None = None

    def __repr__(self) -> str:
        """Represent the GPU resource as a string."""
        return f"GPU(node={self.node}, grank={self.global_rank}, lrank={self.local_rank})"

    @property
    def is_free(self) -> bool:
        """Check if the GPU is free."""
        return self.owner is None

    def allocate_to(self, owner: str) -> GPU:
        """Allocate the GPU to a owner."""
        if self.owner is not None:
            raise ValueError(f"GPU {self} is already allocated to {self.owner}.")

        self.owner = owner

        return self

    def free(self) -> GPU:
        """Free the GPU."""
        if self.owner is None:
            raise ValueError(f"GPU {self} is already free.")

        self.owner = None

        return self


class CannotColocateError(Exception):
    """Exception raised when GPUs cannot be colocated."""

    def __init__(self, message: str) -> None:
        """Initialize the exception."""
        super().__init__(message)
        self.message = message


class NotEnoughGPUsError(Exception):
    """Exception raised when there are not enough GPUs available."""

    def __init__(self, message: str) -> None:
        """Initialize the exception."""
        super().__init__(message)
        self.message = message


class Resource:
    """Cluster resource."""

    def __init__(self, gpus: list[GPU]) -> None:
        """Initialize the resource object."""
        # Create helper data structures and run sanity checks
        # Global ranks should be unique and contiguous
        global_ranks = list(set(gpu.global_rank for gpu in gpus))
        if global_ranks != sorted(range(len(global_ranks))):
            raise ValueError(f"Global ranks should be unique and contiguous. Got {global_ranks}.")

        # Local ranks should be unique and contiguous inside each node
        node_to_gpus: dict[str, list[GPU]] = defaultdict(list)
        for gpu in gpus:
            node_to_gpus[gpu.node].append(gpu)
        for node, node_gpus in node_to_gpus.items():
            local_ranks = list(set(gpu.local_rank for gpu in node_gpus))
            if local_ranks != sorted(range(len(local_ranks))):
                raise ValueError(
                    f"Local ranks should be unique and contiguous inside each node. Got {local_ranks} for {node}.",
                )

        # Ensure nodes are sorted by name and GPUs are sorted by local rank
        self.node_to_gpus: dict[str, list[GPU]] = {}
        for node in sorted(node_to_gpus):
            self.node_to_gpus[node] = sorted(node_to_gpus[node], key=lambda gpu: gpu.local_rank)

        self.gpus = gpus

        self.print_resource_status()

    def num_free_gpus(self, node: str | None = None) -> int:
        """Get the number of free GPUs."""
        if node is None:
            return len([gpu for gpu in self.gpus if gpu.is_free])
        return len([gpu for gpu in self.node_to_gpus[node] if gpu.is_free])

    def allocate(
        self,
        num_gpus: int,
        owner: str,
        must_colocate: bool = True,
        node_selection_policy: Literal["pack", "spread"] = "spread",
    ) -> list[GPU]:
        """Allocate a number of GPUs to a owner.

        GPUs will be allocated on the same node if possible (i.e., colocate).
        Colocation is a hard requirement if `must_colocate` is True, i.e., if
        the GPUs cannot be allocated on the same node, a `CannotColocateError`
        exception will be raised. If `must_colocate` is False, colocation is
        preferred but not required.

        We may have multiple nodes in which we can allocate the GPUs while
        still satisfying colocation. In this case, `node_selection_policy`
        determines which node to select.

        "pack" will try to allocate GPUs on nodes where the owner already has
        GPUs allocated. "spread", on the other hand, will try to avoid
        allocating GPUs on nodes where the owner already has GPUs allocated.

        Args:
            num_gpus: Number of GPUs to allocate
            owner: Owner name
            must_colocate: Whether GPU colocation is required
            node_selection_policy: Node selection policy ("pack" or "spread")

        Raises:
            CannotColocateError: If `must_colocate` is True and GPUs cannot
                be allocated on the same node.
            NotEnoughGPUsError: If there are not enough free GPUs available
                in the cluster.
        """
        logger.info(
            "Request to allocate %d GPUs to %s with must_colocate=%s and node_selection_policy=%s",
            num_gpus,
            owner,
            must_colocate,
            node_selection_policy,
        )

        if num_gpus > (num_free_gpus := self.num_free_gpus()):
            raise NotEnoughGPUsError(
                f"Cannot allocate {num_gpus} GPUs. Only {num_free_gpus} free GPUs available in the cluster.",
            )

        # Nodes with at least one free GPU
        free_nodes: set[str] = set()
        for node, gpus in self.node_to_gpus.items():
            if any(gpu.is_free for gpu in gpus):
                free_nodes.add(node)
        logger.info("Nodes that have at least one free GPU: %s", free_nodes)

        # Order nodes by the number of free GPUs. If nodes have the same number
        # of free GPUs, break ties based on the node selection policy -- by the
        # number of GPUs already allocated to the owner on that node.
        # `heapq` implements a min-heap, so smaller numbers give preference.
        node_priority: list[tuple[int, int, str]] = []
        for node in free_nodes:
            all_gpus = self.node_to_gpus[node]
            num_free_gpus = len([gpu for gpu in all_gpus if gpu.is_free])
            num_gpus_allocated_to_owner = len([gpu for gpu in all_gpus if gpu.owner == owner])
            if node_selection_policy == "pack":
                heapq.heappush(node_priority, (-num_free_gpus, -num_gpus_allocated_to_owner, node))
            elif node_selection_policy == "spread":
                heapq.heappush(node_priority, (-num_free_gpus, num_gpus_allocated_to_owner, node))
            else:
                raise ValueError(f"Unknown node selection policy: {node_selection_policy}")

        # We must find exactly one node with at least `num_gpus` free GPUs.
        # If the head of the heap does not have enough free GPUs, we are out of luck.
        if must_colocate:
            num_free_nodes = -node_priority[0][0]
            if num_free_nodes < num_gpus:
                raise CannotColocateError(
                    f"Cannot allocate {num_gpus} GPUs. No single node has enough free GPUs.",
                )

        # Start allocating GPUs. If `must_colocate`, we have already made sure that
        # the first node in the heap has enough free GPUs to fully satisfy the request.
        num_allocated = 0
        allocated_gpus: list[GPU] = []
        while num_allocated < num_gpus and node_priority:
            _, _, node = heapq.heappop(node_priority)
            gpus = self.node_to_gpus[node]
            gpus = [gpu for gpu in gpus if gpu.is_free][: num_gpus - num_allocated]
            for gpu in gpus:
                allocated_gpus.append(gpu.allocate_to(owner))
                num_allocated += 1
                if num_allocated >= num_gpus:
                    break
            if num_allocated >= num_gpus:
                break

        assert num_allocated == num_gpus
        self.print_resource_status()
        return allocated_gpus

    def print_resource_status(self) -> None:
        """Print the status of the cluster resources."""
        logger.info("Cluster resources: %s", self.gpus)
        logger.info("Cluster global rank status: \n%s", self.visual_repr("global_rank"))
        logger.info("Cluster availability status: \n%s", self.visual_repr("availability"))

    def visual_repr(self, mode: Literal["global_rank", "availability"]) -> str:
        """Construct a visual representation of the cluster resources.

        GPUs are represented by their global rank or availability (owner).

        The visual representation is a 2D grid of GPUs, where each row is a node
        and each column is a GPU. The GPUs are represented by their label ID, and
        the legend shows the mapping from label ID to label name.

        Args:
            mode: The mode of the visual representation.
        """
        if mode == "global_rank":
            grid = []
            max_rank_len = max(len(str(gpu.global_rank)) for gpu in self.gpus)
            max_node_len = max(len(node) for node in self.node_to_gpus)
            for node, gpus in self.node_to_gpus.items():
                row = []
                for gpu in gpus:
                    row.append(f"{gpu.global_rank:>{max_rank_len}}")
                grid.append(f"{node:>{max_node_len}} | " + " ".join(row))
            return "\n".join(grid)

        if mode == "availability":
            labels = sorted(set(gpu.owner for gpu in self.gpus if gpu.owner is not None))
            label_to_id: dict[str | None, str] = {owner: str(i + 1) for i, owner in enumerate(labels)}
            label_to_id[None] = "x"

            grid = []
            max_owner_len = max(len(owner) for owner in label_to_id.values())
            max_node_len = max(len(node) for node in self.node_to_gpus)
            for node, gpus in self.node_to_gpus.items():
                row = []
                for gpu in gpus:
                    owner_id = label_to_id[gpu.owner]
                    row.append(f"{owner_id:>{max_owner_len}}")
                grid.append(f"{node:<{max_node_len}} | " + " ".join(row))
            grid_str = "\n".join(grid)

            legend = "where"
            for owner, owner_id in label_to_id.items():
                owner_name = owner if owner is not None else "Free"
                legend += f"\n  {owner_id:>{max_owner_len}}: {owner_name}"

            return f"{grid_str}\n{legend}"

        raise ValueError(f"Unknown mode: {mode}")
