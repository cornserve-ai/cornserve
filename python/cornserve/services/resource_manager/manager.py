"""Core resource manager class."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from dataclasses import dataclass, field

import grpc
import kubernetes_asyncio.client as kclient
import kubernetes_asyncio.config as kconfig
from opentelemetry import trace

from cornserve import constants
from cornserve.logging import get_logger
from cornserve.services.pb import (
    common_pb2,
    sidecar_pb2,
    sidecar_pb2_grpc,
    task_dispatcher_pb2,
    task_dispatcher_pb2_grpc,
    task_manager_pb2,
    task_manager_pb2_grpc,
)
from cornserve.services.resource import GPU, Resource
from cornserve.services.sidecar.launch import SidecarLaunchInfo
from cornserve.services.utils import to_strict_k8s_name
from cornserve.sidecar.constants import grpc_url_from_rank
from cornserve.task.base import UnitTask
from cornserve.task_executors.profile import UnitTaskProfileManager

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class UnitTaskDeployment:
    """Information about a deployed unit task and its task manager.

    Attributes:
        task: The task being managed
        id: Task manager ID
        url: Task manager URL
    """

    task: UnitTask
    id: str
    url: str


@dataclass
class TaskManagerState:
    """Encapsulates the state of a single task manager.

    This class manages the lifecycle and resources of a single task manager, including its K8s
    resources and gRPC connection. When either spawning or shutting down a task manager, the
    lock object within this class should be acquired.
    """

    id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    resources: list[GPU] | None = None
    pod_name: str | None = None
    service_name: str | None = None
    channel: grpc.aio.Channel | None = None
    stub: task_manager_pb2_grpc.TaskManagerStub | None = None
    deployment: UnitTaskDeployment | None = None

    async def tear_down(self, kube_client: kclient.CoreV1Api) -> None:
        """Clean up all resources associated with this task manager.

        This method is aware of partial failures and will skip cleanup for steps
        that have not been completed due to earlier errors.
        """
        try:
            # Shutdown gRPC
            if self.stub is not None:
                shutdown_req = task_manager_pb2.ShutdownRequest()
                with suppress(grpc.aio.AioRpcError):
                    await self.stub.Shutdown(shutdown_req)
            if self.channel is not None:
                await self.channel.close()

            # Release GPU resources
            if self.resources is not None:
                for gpu in self.resources:
                    gpu.free()

            # Delete K8s pod
            if self.pod_name is not None:
                with suppress(kclient.ApiException):
                    await kube_client.delete_namespaced_pod(
                        name=self.pod_name,
                        namespace=constants.K8S_NAMESPACE,
                    )  # type: ignore
            # Delete K8s service
            if self.service_name is not None:
                with suppress(kclient.ApiException):
                    await kube_client.delete_namespaced_service(
                        name=self.service_name,
                        namespace=constants.K8S_NAMESPACE,
                    )  # type: ignore

        except Exception as e:
            logger.exception(
                "An unexpected exception aborted the cleanup of task manager %s: %s",
                self.id,
                e,
            )
            raise


class ResourceManager:
    """The Resource Manager allocates resources for Task Managers."""

    def __init__(self, api_client: kclient.ApiClient, resource: Resource, sidecar_names: list[str]) -> None:
        """Initialize the ResourceManager."""
        self.api_client = api_client
        self.resource = resource

        self.kube_core_client = kclient.CoreV1Api(api_client)
        self.sidecar_names = sidecar_names

        # Task dispatcher gRPC handles
        self.task_dispatcher_channel = grpc.aio.insecure_channel(constants.K8S_TASK_DISPATCHER_GRPC_URL)
        self.task_dispatcher_stub = task_dispatcher_pb2_grpc.TaskDispatcherStub(self.task_dispatcher_channel)

        # Task state
        self.task_states: dict[str, TaskManagerState] = {}
        self.task_states_lock = asyncio.Lock()

        # Profile manager for GPU allocation
        self.profile_manager = UnitTaskProfileManager()

    @staticmethod
    async def init() -> ResourceManager:
        """Actually initialize the resource manager.

        Spawn the sidecar pods and created GPU objects make up the `Resource` object.
        Initialization has to involve asyncio, so we can't do it in the constructor.
        """
        kconfig.load_incluster_config()
        api_client = kclient.ApiClient()
        core_api = kclient.CoreV1Api(api_client)

        # first find the all the nodes in the cluster
        nodes_response = await core_api.list_node()
        # filter READY nodes
        nodes = []
        for node in nodes_response.items:
            if node.status and node.status.conditions:
                if any(
                    condition.type == "Ready" and condition.status == "True" for condition in node.status.conditions
                ):
                    if (
                        node.status.capacity
                        and "nvidia.com/gpu" in node.status.capacity
                        and int(node.status.capacity["nvidia.com/gpu"]) > 0
                    ):
                        nodes.append(node)
                    else:
                        logger.warning("Node %s has no GPUs, skipping", node.metadata.name)
                else:
                    logger.warning("Node %s is not ready, skipping", node.metadata.name)
            else:
                logger.warning("Node %s has no status or conditions, skipping", node.metadata.name)

        # then for each node, we create num_gpu_per_node sidecars with rank
        coros = []
        gpus = []
        created_pods = []
        try:
            gpu_per_node = {}
            # first query the number of GPUs on each node
            for node in nodes:
                gpu_per_node[node.metadata.name] = int(node.status.capacity["nvidia.com/gpu"])
            world_size = sum(gpu_per_node.values())
            sidecar_rank = 0
            for node in nodes:
                min_node_rank = sidecar_rank
                n_gpu = gpu_per_node[node.metadata.name]
                for j in range(n_gpu):
                    pod = SidecarLaunchInfo.get_pod(
                        node,
                        sidecar_rank,
                        world_size,
                        list(range(min_node_rank, min_node_rank + n_gpu)),
                    )
                    coros.append(core_api.create_namespaced_pod(namespace=constants.K8S_NAMESPACE, body=pod))
                    gpus.append(GPU(node=node.metadata.name, global_rank=sidecar_rank, local_rank=j))
                    sidecar_rank += 1

            spawn_results = await asyncio.gather(*coros, return_exceptions=True)
            failed = 0
            for i, result in enumerate(spawn_results):
                if isinstance(result, BaseException):
                    logger.error("Failed to spawn sidecar pod for GPU %s: %s", gpus[i], result)
                    failed += 1
                else:
                    created_pods.append(result)
                    logger.info("Successfully spawned sidecar pod for GPU %s", gpus[i])

            async def cleanup():
                """Clean up any created pods in case of failure."""
                cleanup_coros = []
                with suppress(kclient.ApiException):
                    for pod in created_pods:
                        cleanup_coros.append(
                            core_api.delete_namespaced_pod(
                                name=pod.metadata.name,
                                namespace=constants.K8S_NAMESPACE,
                            )
                        )
                    await asyncio.gather(*cleanup_coros, return_exceptions=True)

            if failed:
                await cleanup()
                raise RuntimeError(f"Failed to spawn {failed} sidecar pods: {spawn_results}")
            else:

                async def wait_for_online(rank: int) -> None:
                    """Wait for sidecar with `rank` to be online."""
                    while True:
                        try:
                            async with grpc.aio.insecure_channel(grpc_url_from_rank(rank)) as channel:
                                req = sidecar_pb2.CheckHealthRequest()
                                stub = sidecar_pb2_grpc.SidecarStub(channel)
                                res = await stub.CheckHealth(req)
                                if res.status == sidecar_pb2.HealthStatus.HEALTH_ALL_GOOD:
                                    logger.info("Sidecar %d is online", rank)
                                    return
                                await asyncio.sleep(1)
                        except Exception:
                            await asyncio.sleep(1)

                coros = [wait_for_online(gpu.global_rank) for gpu in gpus]
                try:
                    async with asyncio.timeout(SidecarLaunchInfo.DEFAULT_LAUNCH_TIMEOUT):
                        await asyncio.gather(*coros)
                except TimeoutError as e:
                    logger.error("Timed out waiting for sidecars to come online")
                    await cleanup()
                    raise RuntimeError(f"Failed to spawn {failed} sidecar pods: {spawn_results}") from e
                logger.info("All sidecars are online")

            resource = Resource(gpus=gpus)
            return ResourceManager(
                api_client=api_client,
                resource=resource,
                sidecar_names=[pod.metadata.name for pod in created_pods],
            )
        except Exception as e:
            logger.error("Error during resource initialization: %s", str(e))
            raise

    @tracer.start_as_current_span("ResourceManager.scale_up_unit_task")
    async def scale_up_unit_task(self, task: UnitTask, num_gpus: int) -> None:
        """Scale up a unit task by allocating additional GPUs to the task manager."""
        assert num_gpus > 0, "Number of GPUs to scale up must be positive"
        logger.info("Scaling up unit task %s by %d GPUs", task, num_gpus)

        span = trace.get_current_span()
        span.set_attribute("resource_manager.scale_up_unit_task.task", str(task))
        span.set_attribute("resource_manager.scale_up_unit_task.num_gpus", num_gpus)

        # Check if the number of GPUs to scale up is valid
        profile = self.profile_manager.get_profile(task)
        if num_gpus < min(profile.keys()):
            logger.warning(
                "Requested %d GPUs to scale up task %s, but minimum required is %d GPUs according to the profile: %s",
                num_gpus,
                task,
                min(profile.keys()),
                profile,
            )
            raise ValueError(
                f"Cannot scale up task {task} by {num_gpus} GPUs. "
                f"Minimum required is {min(profile.keys())} GPUs according to the profile."
            )

        # Find the task manager state for this task
        task_state = None
        async with self.task_states_lock:
            for state in self.task_states.values():
                if state.deployment and state.deployment.task.is_equivalent_to(task):
                    task_state = state

            if task_state is None:
                logger.info("Task %s is not running, returning immediately", task)
                return

            # TODO: decide GPU placement strategy & preference
            resources = []
            allocate_granularity = min(profile.keys())
            remaining_gpus = num_gpus % allocate_granularity
            # sanity check
            if remaining_gpus != 0 and remaining_gpus not in profile:
                raise ValueError(
                    f"Cannot scale up task {task} by {num_gpus} GPUs. "
                    f"There will be {remaining_gpus} GPUs left unallocated "
                    f"according to the profile: {profile.keys()}. ",
                )
            for _ in range(num_gpus // allocate_granularity):
                # here we make sure colocation
                batched_resources = self.resource.allocate(
                    num_gpus=allocate_granularity,
                    owner=task_state.id,
                )
                resources.extend(batched_resources)
            for _ in range(remaining_gpus):
                # allocate the remaining GPUs one by one
                resources.append(self.resource.allocate(num_gpus=1, owner=task_state.id)[0])

        assert task_state.stub is not None, "Task manager stub is not initialized"
        async with task_state.lock:
            try:
                gpus = [
                    task_manager_pb2.GPUResource(
                        action=task_manager_pb2.ResourceAction.ADD,
                        node_id=gpu.node,
                        global_rank=gpu.global_rank,
                        local_rank=gpu.local_rank,
                    )
                    for gpu in resources
                ]
                update_resource_req = task_manager_pb2.UpdateResourcesRequest(task_manager_id=task_state.id, gpus=gpus)
                response = await task_state.stub.UpdateResources(update_resource_req)
                if response.status != common_pb2.Status.STATUS_OK:
                    raise RuntimeError(
                        f"UpdateResources gRPC error when trying to scale up task manager: {response}",
                    )
                # Add the new GPUs to the task state resources
                async with self.task_states_lock:
                    if task_state.resources is None:
                        task_state.resources = []
                    task_state.resources.extend(resources)
            except Exception as e:
                async with self.task_states_lock:
                    for gpu in resources:
                        gpu.free()
                logger.exception("Failed to scale up task %s by %d GPUs: %s", task, num_gpus, e)
                raise RuntimeError(f"Failed to scale up task {task} by {num_gpus} GPUs: {e}") from e

    @tracer.start_as_current_span("ResourceManager.scale_down_unit_task")
    async def scale_down_unit_task(self, task: UnitTask, num_gpus: int) -> None:
        """Scale down a unit task by removing GPUs from the task manager."""
        assert num_gpus > 0, "Number of GPUs to scale down must be positive"
        logger.info("Scaling down unit task %s by %d GPUs", task, num_gpus)

        span = trace.get_current_span()
        span.set_attribute("resource_manager.scale_down_unit_task.task", str(task))
        span.set_attribute("resource_manager.scale_down_unit_task.num_gpus", num_gpus)

        # Find the task manager state for this task
        task_state = None
        async with self.task_states_lock:
            for state in self.task_states.values():
                if state.deployment and state.deployment.task == task:
                    task_state = state

            if task_state is None:
                logger.info("Task %s is not running, returning immediately", task)
                return

        assert task_state.stub is not None, "Task manager stub is not initialized"
        # check there are enough GPUs to scale down
        if task_state.resources is None or len(task_state.resources) < num_gpus:
            raise RuntimeError(
                f"{task} has only "
                f"{len(task_state.resources) if task_state.resources else 0} GPUs, "
                f"cannot scale down by {num_gpus}"
            )
        async with task_state.lock:
            try:
                # TODO: decide GPU placement strategy & preference
                gpus_to_remove = task_state.resources[:num_gpus]
                gpus = [
                    task_manager_pb2.GPUResource(
                        action=task_manager_pb2.ResourceAction.REMOVE,
                        node_id=gpu.node,
                        global_rank=gpu.global_rank,
                        local_rank=gpu.local_rank,
                    )
                    for gpu in gpus_to_remove
                ]
                update_resource_req = task_manager_pb2.UpdateResourcesRequest(task_manager_id=task_state.id, gpus=gpus)
                response = await task_state.stub.UpdateResources(update_resource_req)
                if response.status != common_pb2.Status.STATUS_OK:
                    raise RuntimeError(
                        f"UpdateResources gRPC error when trying to scale down task manager: {response}",
                    )
                # Remove the GPUs from the task state resources
                async with self.task_states_lock:
                    task_state.resources = task_state.resources[num_gpus:]
                    for gpu in gpus_to_remove:
                        gpu.free()
            except Exception as e:
                logger.exception("Failed to scale down task %s by %d GPUs: %s", task, num_gpus, e)
                raise RuntimeError(f"Failed to scale down task {task} by {num_gpus} GPUs: {e}") from e

    @tracer.start_as_current_span("ResourceManager.deploy_unit_task")
    async def deploy_unit_task(self, task: UnitTask) -> None:
        """Deploy a unit task by spawning its task manager if needed.

        If this task is already running, this method is a no-op.

        Args:
            task: The task to be deployed.
        """
        logger.info("Deploying unit task %s", task)

        span = trace.get_current_span()
        span.set_attribute("resource_manager.deploy_unit_task.task", str(task))

        async with self.task_states_lock:
            # See if the task is already running
            for state in self.task_states.values():
                if state.deployment and state.deployment.task.is_equivalent_to(task):
                    logger.info("Task %s is already running, returning immediately", task)
                    return

            # Create a new task manager state
            task_manager_id = f"{task.make_name()}-{uuid.uuid4().hex[:8]}"
            state = TaskManagerState(id=task_manager_id)
            self.task_states[task_manager_id] = state

        # Spawn task manager with state-specific lock
        async with state.lock:
            try:
                deployment = await self._spawn_task_manager(task, state)
                state.deployment = deployment
                logger.info("Successfully deployed unit task %s with task manager %s", task, deployment)
            except Exception as e:
                logger.error("Failed to spawn task manager for %s: %s", task, e)
                async with self.task_states_lock:
                    self.task_states.pop(task_manager_id, None)
                raise RuntimeError(f"Failed to spawn task manager for {task}: {e}") from e

        # Notify the task dispatcher of the new task and task manager
        task_manager_info = task_dispatcher_pb2.NotifyUnitTaskDeploymentRequest(
            task=task.to_pb(),
            task_manager=task_dispatcher_pb2.TaskManagerDeployment(
                url=deployment.url,
            ),
        )
        try:
            await self.task_dispatcher_stub.NotifyUnitTaskDeployment(task_manager_info)
        except grpc.aio.AioRpcError as e:
            logger.error(
                "Failed to notify task dispatcher of new task %s and task manager %s: %s",
                task,
                deployment,
                e,
            )
            # Clean up the task manager since notification failed
            async with self.task_states_lock:
                if task_state := self.task_states.pop(task_manager_id, None):
                    logger.info("Cleaning up task manager %s", task_state.id)
                    await task_state.tear_down(self.kube_core_client)
            raise RuntimeError(f"Failed to notify task dispatcher of new task {task}: {e}") from e

    @tracer.start_as_current_span("ResourceManager.teardown_unit_task")
    async def teardown_unit_task(self, task: UnitTask) -> None:
        """Tear down a unit task by shutting down its task manager if needed.

        If this task is not running, this method is a no-op.

        Args:
            task: The unit task to be torn down.
        """
        logger.info("Tearing down unit task %s", task)

        span = trace.get_current_span()
        span.set_attribute("resource_manager.teardown_unit_task.task", str(task))

        # Find the task manager state for this task
        task_state = None
        async with self.task_states_lock:
            for state_id, state in self.task_states.items():
                if state.deployment and state.deployment.task == task:
                    task_state = state
                    # Remove from task_states dict but don't cleanup yet
                    self.task_states.pop(state_id)
                    break

            if task_state is None:
                logger.info("Task %s is not running, returning immediately", task)
                return

        # First notify the task dispatcher of the removed task
        task_info = task_dispatcher_pb2.NotifyUnitTaskTeardownRequest(task=task.to_pb())
        try:
            await self.task_dispatcher_stub.NotifyUnitTaskTeardown(task_info)
        except Exception as e:
            # Do not re-raise the exception but rather continue with the shutdown
            logger.exception("Failed to notify task dispatcher of removed task %s: %s", task, e)

        # Clean up the task manager state with its individual lock
        async with task_state.lock:
            try:
                await task_state.tear_down(self.kube_core_client)
            except Exception as e:
                logger.error("Failed to clean up task manager for %s: %s", task, e)
                raise RuntimeError(f"Failed to clean up task manager for {task}: {e}") from e

    async def healthcheck(self) -> tuple[bool, list[tuple[UnitTask, bool]]]:
        """Check the health of all task managers.

        We intentionally do not hold any locks while performing the health check,
        because we don't want to block other operations. It's fine even if a new
        task manager is missed or an error arises from a task manager that is being
        shut down.

        Returns:
            Tuple of overall_healthy and list of (task_manager_id, healthy)
        """
        logger.info("Performing health check of all task managers")

        task_manager_statuses: list[tuple[UnitTask, bool]] = []
        all_healthy = True

        task_manager_ids = []
        tasks = []
        check_tasks = []
        for task_manager_id, task_manager in self.task_states.items():
            if task_manager.stub is None or task_manager.deployment is None:
                continue
            task_manager_ids.append(task_manager_id)
            tasks.append(task_manager.deployment.task)
            check_tasks.append(task_manager.stub.Healthcheck(task_manager_pb2.HealthcheckRequest(), timeout=1.0))

        # Wait for all health checks to complete
        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        for task_manager_id, task, result in zip(task_manager_ids, tasks, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Health check failed for task manager %s: %s", task_manager_id, str(result))
                task_manager_statuses.append((task, False))
                all_healthy = False
            else:
                # Check if task manager is healthy (status OK = 0)
                is_healthy = result.status == common_pb2.Status.STATUS_OK
                if not is_healthy:
                    all_healthy = False
                # TODO(J1): Task executor details should be propagated up
                task_manager_statuses.append((task_manager_id, is_healthy))

        return all_healthy, task_manager_statuses

    async def shutdown(self) -> None:
        """Shutdown the ResourceManager."""
        async with self.task_states_lock:
            coros = []
            for task_state in self.task_states.values():
                coros.append(task_state.tear_down(self.kube_core_client))
            for name in self.sidecar_names:
                coros.append(
                    self.kube_core_client.delete_namespaced_pod(
                        name=name,
                        namespace=constants.K8S_NAMESPACE,
                    )
                )
            results = await asyncio.gather(*coros, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                logger.error("Error occured during shutdown: %s", result)

        await self.api_client.close()
        await self.task_dispatcher_channel.close()

    @tracer.start_as_current_span("ResourceManager._spawn_task_manager")
    async def _spawn_task_manager(self, task: UnitTask, state: TaskManagerState) -> UnitTaskDeployment:
        """Spawn a new task manager.

        If anything goes wrong, side effects are cleaned up and an exception is raised.

        Args:
            task: The task to be deployed.
            state: The TaskManagerState object to store the task manager's state.
        """
        logger.info("Spawning task manager %s for %s", state.id, task)
        span = trace.get_current_span()
        span.set_attribute("resource_manager._spawn_task_manager.task_manager_id", state.id)

        try:
            # Get GPU requirements from profile
            profile = self.profile_manager.get_profile(task)

            # Start off the task manager with the minimum number of GPUs required
            num_gpus = min(profile.keys())

            logger.info("Allocating %d GPUs for task %s based on profile", num_gpus, task)

            # Allocate resource starter pack for the task manager
            state.resources = self.resource.allocate(num_gpus=num_gpus, owner=state.id)
            span.set_attribute(
                "resource_manager._spawn_task_manager.gpu_global_ranks",
                [gpu.global_rank for gpu in state.resources],
            )

            # Create a new task manager pod and service
            state.pod_name = f"tm-{state.id}".lower()
            state.service_name = to_strict_k8s_name(state.pod_name)
            port = 50051

            pod = kclient.V1Pod(
                metadata=kclient.V1ObjectMeta(
                    name=state.pod_name,
                    labels={
                        "app": "task-manager",
                        "task-manager-id": state.id,
                    },
                ),
                spec=kclient.V1PodSpec(
                    service_account_name="task-manager",
                    containers=[
                        kclient.V1Container(
                            name="task-manager",
                            image=constants.CONTAINER_IMAGE_TASK_MANAGER,
                            image_pull_policy=constants.CONTAINER_IMAGE_PULL_POLICY,
                            ports=[kclient.V1ContainerPort(container_port=port, name="grpc")],
                            env_from=[
                                kclient.V1EnvFromSource(
                                    config_map_ref=kclient.V1ConfigMapEnvSource(
                                        name=constants.K8S_CORNSERVE_CONFIG_MAP_NAME,
                                    )
                                ),
                            ],
                            volume_mounts=[
                                kclient.V1VolumeMount(
                                    name="cornserve-profiles",
                                    mount_path=constants.UNIT_TASK_PROFILES_DIR,
                                    read_only=True,
                                ),
                            ],
                        )
                    ],
                    volumes=[
                        kclient.V1Volume(
                            name="cornserve-profiles",
                            config_map=kclient.V1ConfigMapVolumeSource(
                                name=constants.K8S_UNIT_TASK_PROFILES_CONFIG_MAP_NAME,
                                optional=True,
                            ),
                        ),
                    ],
                ),
            )
            service = kclient.V1Service(
                metadata=kclient.V1ObjectMeta(
                    name=state.service_name,
                    labels={
                        "app": "task-manager",
                        "task-manager-id": state.id,
                    },
                ),
                spec=kclient.V1ServiceSpec(
                    selector={
                        "app": "task-manager",
                        "task-manager-id": state.id,
                    },
                    ports=[kclient.V1ServicePort(port=port, target_port="grpc")],
                ),
            )

            with tracer.start_as_current_span("ResourceManager._spawn_task_manager.create_pod"):
                await self.kube_core_client.create_namespaced_pod(
                    namespace=constants.K8S_NAMESPACE,
                    body=pod,
                )  # type: ignore

            with tracer.start_as_current_span("ResourceManager._spawn_task_manager.create_service"):
                await self.kube_core_client.create_namespaced_service(
                    namespace=constants.K8S_NAMESPACE,
                    body=service,
                )  # type: ignore

            logger.info("Created task manager pod %s and service %s", state.pod_name, state.service_name)

            # Connect to the task manager gRPC server to initialize it
            state.channel = grpc.aio.insecure_channel(f"{state.service_name}:{port}")
            state.stub = task_manager_pb2_grpc.TaskManagerStub(state.channel)

            # Initialize the task manager by providing it with the task it will manage
            # and an initial set of GPU resources to work with.
            with tracer.start_as_current_span("ResourceManager._spawn_task_manager.register_task"):
                register_task_req = task_manager_pb2.RegisterTaskRequest(
                    task_manager_id=state.id,
                    task=task.to_pb(),
                    gpus=[
                        task_manager_pb2.GPUResource(
                            action=task_manager_pb2.ResourceAction.ADD,
                            node_id=gpu.node,
                            global_rank=gpu.global_rank,
                            local_rank=gpu.local_rank,
                        )
                        for gpu in state.resources
                    ],
                )
                response: task_manager_pb2.RegisterTaskResponse = await state.stub.RegisterTask(
                    register_task_req, wait_for_ready=True
                )
                if response.status != common_pb2.Status.STATUS_OK:
                    raise RuntimeError(f"Failed to register task manager: {response}")

        except Exception as e:
            logger.exception("Failed to spawn task manager: %s", e)
            await state.tear_down(self.kube_core_client)
            raise RuntimeError(f"Failed to initialize spawned task manager for {task}: {e}") from e

        return UnitTaskDeployment(task=task, id=state.id, url=f"{state.service_name}:{port}")
