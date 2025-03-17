"""TaskManager class."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import suppress
import uuid

import httpx
import kubernetes_asyncio.client as kclient
import kubernetes_asyncio.config as kconfig

from cornserve import constants
from pydantic import ValidationError

from cornserve.services.task_manager.models import (
    TaskManagerType,
    TaskManagerConfig,
    EncoderConfig,
    LLMConfig,
)
from cornserve.services.resource_manager.resource import GPU
from cornserve.task_executors.launch import TaskExecutorLaunchInfo
from cornserve.logging import get_logger

logger = get_logger(__name__)


class TaskManager:
    """Task manager abstract base class."""

    def __init__(self, id: str, config: TaskManagerConfig, launch_info: TaskExecutorLaunchInfo) -> None:
        """Initialize the TaskManager."""
        self.id = id
        self.config = config
        self.launch_info = launch_info
        self.gpus: list[GPU] = []

        self.lock = asyncio.Lock()
        self.executor_urls: dict[str, str] = {}
        self.executor_gpus: dict[str, list[GPU]] = defaultdict(list)
        self.executor_pod_names: dict[str, str] = {}
        self.executor_service_names: dict[str, str] = {}

        self.http_client = httpx.AsyncClient()

        kconfig.load_incluster_config()
        self.k8s_client = kclient.ApiClient()
        self.core_client = kclient.CoreV1Api(api_client=self.k8s_client)

        # Config variables
        self.task_executor_healthy_timeout = 5 * 60.0

    @staticmethod
    async def init(
        id: str,
        task_type: TaskManagerType,
        config_str: str,
        gpus: list[GPU],
    ) -> TaskManager:
        """Initialize the designated task manager.

        Args:
            id: Unique identifier for the task manager
            task_type: Type of task manager to create
            config_str: JSON string containing task configuration
            gpus: List of initial GPU resources allocated to the task manager
        """
        try:
            manager_cls: type[TaskManager]
            if task_type == TaskManagerType.ENCODER:
                config = EncoderConfig.model_validate_json(config_str)
                manager_cls = EncoderTaskManager
            elif task_type == TaskManagerType.LLM:
                config = LLMConfig.model_validate_json(config_str)
                manager_cls = LLMTaskManager
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except ValidationError as e:
            raise ValueError(f"Invalid config for {task_type}: {e}") from e

        launch_info = TaskExecutorLaunchInfo.from_task_manager_config(config)
        manager = manager_cls(id, config, launch_info)

        # Add initial set of GPUs
        await manager.update_resources(add_gpus=gpus)

        return manager

    async def update_resources(
        self,
        add_gpus: list[GPU] | None = None,
        remove_gpus: list[GPU] | None = None,
    ) -> None:
        """Update the resources allocated to the task manager.

        The default implementation will first kill all task executors that
        were using GPUs part of the `remove_gpus` list and reap the GPUs.
        Then, together with the `add_gpus` list, it will launch new task
        executors and allocate the GPUs to them. Each task executor will
        just use one GPU.
        """
        logger.info(
            "Updating resources for task manager %s: add_gpus=%s, remove_gpus=%s",
            self.id,
            add_gpus,
            remove_gpus,
        )

        add_gpus = add_gpus or []
        remove_gpus = remove_gpus or []

        async with self.lock:
            # Sanity check
            if not all(gpu in self.gpus for gpu in remove_gpus):
                raise ValueError("Cannot remove GPUs that are not allocated")

            if any(gpu in self.gpus for gpu in add_gpus):
                raise ValueError("Cannot add GPUs that are already allocated")

            # Update GPU list
            self.gpus = [gpu for gpu in self.gpus if gpu not in remove_gpus]
            self.gpus.extend(add_gpus)

            # First, kill executors that were using the removed GPUs
            to_kill = []
            for executor_id, gpus in self.executor_gpus.items():
                executor_removed_gpu = []
                for gpu in gpus:
                    if gpu in remove_gpus:
                        executor_removed_gpu.append(gpu)
                if executor_removed_gpu:
                    to_kill.append(executor_id)
                    logger.info(
                        "Killing task executor %s due to removal of GPU %s",
                        executor_id,
                        executor_removed_gpu,
                    )

            kill_results = await asyncio.gather(
                *[self._kill_executor(executor_id) for executor_id in to_kill],
                return_exceptions=True,
            )

            # Check for errors in killing
            failed = 0
            for result in kill_results:
                if isinstance(result, BaseException):
                    logger.error("Failed to kill task executor: %s", result)
                    failed += 1

            if failed:
                logger.error("Failed to kill %d task executors. Spawning new ones with free GPUs anyway.", failed)

            # GPUs are marked as free inside `_kill_executor` after the executor is killed
            free_gpus = [gpu for gpu in self.gpus if gpu.is_free]

            # Spawn new executors with free GPUs
            spawn_results = await asyncio.gather(
                *[self._spawn_executor([gpu]) for gpu in free_gpus],
                return_exceptions=True,
            )

            # Check for errors in spawning
            failed = 0
            for result in spawn_results:
                if isinstance(result, BaseException):
                    logger.error("Failed to spawn task executor: %s", result)
                    failed += 1

            if failed:
                logger.error("Failed to spawn %d task executors.", failed)
                raise RuntimeError("Failed to spawn task executors")

    async def get_route(self, app_id: str, request_id: str, routing_hint: str) -> tuple[str, list[int]]:
        """Get the URL to the task executor for a request.

        The default implementation implemets sticky routing by hashing
        the request ID to an integer and using that to index into the
        task executor list.

        Args:
            app_id: Unique identifier for the application
            request_id: Unique identifier for the request
            routing_hint: Arbitrary string to hint the routing decision

        Returns:
            URL to the task executor to handle the request and a list of
            sidecar ranks.
        """
        logger.info("Routing request %s for app %s with routing hint %s", request_id, app_id, routing_hint)

        async with self.lock:
            urls = list(self.executor_urls.values())
            gpus = list(self.executor_gpus.values())

        index = hash(request_id) % len(urls)
        route = urls[index]
        sidecar_ranks = [gpu.global_rank for gpu in gpus[index]]

        logger.info("Routing request %s to %s (sidecars %s)", request_id, route, sidecar_ranks)

        return (route, sidecar_ranks)

    async def _do_healthcheck(self, executor_url: str, timeout: float = 1.0) -> bool:
        """Perform healthcheck on a single task executor.

        This method assumes that the healthcheck endpoint is `GET /health`.
        """
        try:
            response = await self.http_client.get(f"{executor_url}/health", timeout=timeout)
        except Exception as e:
            logger.error("Failed to healthcheck %s: %s", executor_url, e)
            return False
        return response.status_code == 200

    async def healthcheck(self) -> dict[str, bool]:
        """Perform healthcheck on all task executors.

        The default implementation performs healthcheck on the task
        executors assuming the healthcheck endpoint is `GET /health`.

        Returns:
            Dict of task executor IDs to whether they are healthy
        """
        async with self.lock:
            executor_ids = list(self.executor_urls)
            executor_urls = list(self.executor_urls.values())

        tasks = [self._do_healthcheck(url) for url in executor_urls]
        responses = await asyncio.gather(*tasks)
        return dict(zip(executor_ids, responses, strict=True))

    async def shutdown(self) -> None:
        """Shutdown the task manager."""
        # Kill all task executors
        async with self.lock:
            executor_ids = list(self.executor_urls)
            kill_results = await asyncio.gather(
                *[self._kill_executor(executor_id) for executor_id in executor_ids],
                return_exceptions=True,
            )

        if any(r is not None for r in kill_results):
            logger.error(
                "Failed to kill some task executors: %s",
                [r for r in kill_results if r is not None],
            )

        await self.http_client.aclose()
        await self.k8s_client.close()

        logger.info("Task manager %s shut down", self.id)

    async def _spawn_executor(self, gpus: list[GPU]) -> None:
        """Spawn a new task executor with the given GPU and wait for it to become available.

        This method assumes that the lock is held by the caller.

        The default implementation requires that all GPUs be on the same node.

        Args:
            gpus: GPU resources to allocate to the executor
        """
        logger.info("Spawning task executor with GPUs: %s", gpus)

        # Ensure all GPUs are on the same node
        node_names = {gpu.node for gpu in gpus}
        if len(node_names) != 1:
            raise ValueError("All GPUs must be on the same node")
        node_name = node_names.pop()

        executor_id = self.launch_info.get_executor_name().lower()
        executor_id = "-".join([executor_id, *(f"{gpu.global_rank}" for gpu in gpus)])
        pod_name = f"task-executor-{executor_id}"
        service_name = f"task-executor-{executor_id}"
        port = 8000

        # Labels cannot be longer than 63 characters, but executor IDs can be longer.
        # Therefore, we to generate a random label for the executor ID.
        executor_id_label = str(uuid.uuid4())

        # Create the pod spec
        pod = kclient.V1Pod(
            metadata=kclient.V1ObjectMeta(
                name=pod_name,
                labels={
                    "app": "task-executor",
                    "executor-id": executor_id_label,
                    "task-type": self.config.type.lower(),
                },
            ),
            spec=kclient.V1PodSpec(
                containers=[
                    kclient.V1Container(
                        name="task-executor",
                        image=self.launch_info.get_container_image(),
                        image_pull_policy="Always",
                        args=self.launch_info.get_container_args(gpus, port),
                        ports=[kclient.V1ContainerPort(container_port=port, name="http")],
                        resources=kclient.V1ResourceRequirements(
                            limits={
                                "nvidia.com/gpu": len(gpus),
                            }
                        ),
                        env=[
                            kclient.V1EnvVar(
                                name="NVIDIA_VISIBLE_DEVICES",
                                value=",".join(str(gpu.local_rank) for gpu in gpus),
                            ),
                        ],
                        volume_mounts=[
                            kclient.V1VolumeMount(
                                name=name,
                                mount_path=container_path,
                            )
                            for name, _, container_path in self.launch_info.get_container_volumes()
                        ],
                    )
                ],
                volumes=[
                    kclient.V1Volume(
                        name=name,
                        host_path=kclient.V1HostPathVolumeSource(path=host_path),
                    )
                    for name, host_path, _ in self.launch_info.get_container_volumes()
                ],
                node_name=node_name,
                host_ipc=True,
            ),
        )

        # Create the service spec
        service = kclient.V1Service(
            metadata=kclient.V1ObjectMeta(
                name=service_name,
                labels={
                    "app": "task-executor",
                    "executor-id": executor_id_label,
                },
            ),
            spec=kclient.V1ServiceSpec(
                selector={
                    "app": "task-executor",
                    "executor-id": executor_id_label,
                },
                ports=[kclient.V1ServicePort(port=port, target_port="http")],
            ),
        )

        try:
            # Create pod and service
            await self.core_client.create_namespaced_pod(
                namespace=constants.K8S_NAMESPACE,
                body=pod,
            )  # type: ignore

            self.executor_pod_names[executor_id] = pod_name

            await self.core_client.create_namespaced_service(
                namespace=constants.K8S_NAMESPACE,
                body=service,
            )  # type: ignore

            executor_url = f"http://{service_name}:{port}"
            self.executor_service_names[executor_id] = service_name
            self.executor_urls[executor_id] = executor_url

            # Wait for the task executor to become available
            healthy_deadline = asyncio.get_event_loop().time() + self.task_executor_healthy_timeout
            while True:
                if asyncio.get_event_loop().time() > healthy_deadline:
                    raise TimeoutError("Timed out waiting for task executor to become healthy")

                if await self._do_healthcheck(executor_url, timeout=1.0):
                    break

                await asyncio.sleep(0.5)

            # Allocate the GPUs to the executor
            for gpu in gpus:
                self.executor_gpus[executor_id].append(gpu.allocate_to(executor_id))

        except Exception as e:
            logger.exception("Failed to spawn task executor: %s", e)
            await self._kill_executor(executor_id)
            raise

    async def _kill_executor(self, executor_id: str) -> None:
        """Kill a task executor and clean up its resources and wait until it is gone.

        This method assumes that the lock is held by the caller.

        Args:
            executor_id: ID of the executor to kill
        """
        logger.info("Killing task executor %s", executor_id)

        try:
            if pod_name := self.executor_pod_names.pop(executor_id, None):
                with suppress(kclient.ApiException):
                    await self.core_client.delete_namespaced_pod(
                        name=pod_name,
                        namespace=constants.K8S_NAMESPACE,
                    )  # type: ignore

            if service_name := self.executor_service_names.pop(executor_id, None):
                with suppress(kclient.ApiException):
                    await self.core_client.delete_namespaced_service(
                        name=service_name,
                        namespace=constants.K8S_NAMESPACE,
                    )  # type: ignore

            self.executor_urls.pop(executor_id, None)

            # Wait until the pod and service are gone
            if pod_name is not None:
                while True:
                    try:
                        await self.core_client.read_namespaced_pod(
                            name=pod_name,
                            namespace=constants.K8S_NAMESPACE,
                        )  # type: ignore
                    except kclient.ApiException as e:
                        if e.status == 404:
                            break
                    await asyncio.sleep(1)

            if service_name is not None:
                while True:
                    try:
                        await self.core_client.read_namespaced_service(
                            name=service_name,
                            namespace=constants.K8S_NAMESPACE,
                        )  # type: ignore
                    except kclient.ApiException as e:
                        if e.status == 404:
                            break
                    await asyncio.sleep(1)

            # Free GPUs as the final step.
            # Executors that failed to terminate properly will not free GPUs,
            # because they might still be in use or there may be processes
            # occupying memory, likely leading to performance issues or failure
            # later when those GPUs are occupied by new executors.
            if gpus := self.executor_gpus.pop(executor_id, None):
                for gpu in gpus:
                    gpu.free()

        except Exception as e:
            logger.exception("Failed to kill task executor %s: %s", executor_id, e)
            raise


class EncoderTaskManager(TaskManager):
    """Task manager for multimodal data encoder tasks."""


class LLMTaskManager(TaskManager):
    """Task manager for LLM text generation tasks."""
