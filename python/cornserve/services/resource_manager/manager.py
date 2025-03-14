"""Core resource manager class."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import suppress

import grpc
import kubernetes_asyncio.config as kconfig
import kubernetes_asyncio.client as kclient

from cornserve import constants
from cornserve.logging import get_logger
from cornserve.frontend.tasks import Task
from cornserve.services.task_manager.models import TaskManagerConfig
from cornserve.services.pb import (
    task_manager_pb2,
    task_manager_pb2_grpc,
    common_pb2,
)

from .resource import GPU, Resource

logger = get_logger(__name__)


class ResourceManager:
    """The Resource Manager allocates resources for Task Managers."""

    def __init__(
        self,
        api_client: kclient.ApiClient,
        resource: Resource,
    ) -> None:
        """Initialize the ResourceManager."""
        self.api_client = api_client
        self.resource = resource

        self.kube_core_client = kclient.CoreV1Api(api_client)

        # App state
        self.app_lock = asyncio.Lock()
        self.app_task_manager_configs: dict[str, list[TaskManagerConfig]] = {}
        self.app_task_manager_ids: dict[str, list[str]] = {}

        # Task manager state
        self.task_manager_lock = asyncio.Lock()
        self.task_manager_resources: dict[str, list[GPU]] = {}
        self.task_manager_pods: dict[str, str] = {}
        self.task_manager_services: dict[str, str] = {}
        self.task_manager_channels: dict[str, grpc.aio.Channel] = {}
        self.task_manager_stubs: dict[str, task_manager_pb2_grpc.TaskManagerStub] = {}

    @staticmethod
    async def init() -> ResourceManager:
        """Actually initialize the resource manager.

        Initialization has to involve asyncio, so we can't do it in the constructor.
        """
        kconfig.load_incluster_config()
        api_client = kclient.ApiClient()
        core_api = kclient.CoreV1Api(api_client)
        apps_api = kclient.AppsV1Api(api_client)

        # Wait until the sidecars are all ready
        while True:
            await asyncio.sleep(1)
            sidecar_set = await apps_api.read_namespaced_stateful_set(  # type: ignore
                name="sidecar",
                namespace=constants.K8S_NAMESPACE,
            )
            if sidecar_set.status.ready_replicas == sidecar_set.spec.replicas:  # type: ignore
                break
            logger.info("Waiting for sidecar pods to be ready...")
        logger.info("All sidecar %d pods are ready.", sidecar_set.status.ready_replicas)  # type: ignore

        # Discover the sidecar pods
        label_selector = ",".join(
            f"{key}={value}"
            for key, value in sidecar_set.spec.selector.match_labels.items()  # type: ignore
        )
        sidecar_pod_list = await core_api.list_namespaced_pod(
            namespace=constants.K8S_NAMESPACE,
            label_selector=label_selector,
        )
        sidecar_pods = sidecar_pod_list.items

        # Construct cluster resource object
        node_to_pods: dict[str, list[kclient.V1Pod]] = defaultdict(list)
        for pod in sidecar_pods:
            node = pod.spec.node_name
            node_to_pods[node].append(pod)
        gpus = []
        for node, pods in node_to_pods.items():
            for local_rank, pod in enumerate(sorted(pods, key=lambda p: p.metadata.name)):  # type: ignore
                global_rank = int(pod.metadata.name.split("-")[-1])  # type: ignore
                gpu = GPU(node=node, global_rank=global_rank, local_rank=local_rank)
                gpus.append(gpu)
        resource = Resource(gpus=gpus)

        return ResourceManager(api_client=api_client, resource=resource)

    async def reconcile_new_app(self, app_id: str, tasks: list[Task]) -> None:
        """Reconcile new app by spawning task managers if needed."""
        logger.info("Reconcile new app %s with tasks %s", app_id, tasks)

        if app_id in self.app_task_manager_configs:
            raise ValueError(f"App {app_id} already registered")

        # Construct task manager config objects
        task_manager_configs: list[TaskManagerConfig] = []
        for task in tasks:
            task_manager_configs.extend(TaskManagerConfig.from_task(task))
        logger.info("Task manager configs: %s", task_manager_configs)

        # A task manager can be shared by multiple apps.
        # We only spawn task managers that are not already running.
        await self.app_lock.acquire()
        try:
            # Spawn task managers for the new app
            all_task_manager_configs = set(self.app_task_manager_configs.values())
            spawn_task_manager_configs = []
            coros = []
            for task_manager_config in task_manager_configs:
                if task_manager_config not in all_task_manager_configs:
                    spawn_task_manager_configs.append(task_manager_config)
                    coros.append(self._spawn_task_manager(task_manager_config))
            logger.info("Spawning task managers: %s", spawn_task_manager_configs)
            spawn_results = await asyncio.gather(*coros, return_exceptions=True)

            # Check for errors
            failed = 0
            task_manager_ids: list[str] = []
            for i, task_manager_id in enumerate(spawn_results):
                if isinstance(task_manager_id, BaseException):
                    logger.error(
                        "Failed to spawn task manager %s: %s",
                        task_manager_configs[i],
                        task_manager_id,
                    )
                    failed += 1
                else:
                    logger.info(
                        "Successfully spawned task manager %s: %s",
                        task_manager_configs[i],
                        task_manager_id,
                    )
                    task_manager_ids.append(task_manager_id)
            if failed:
                raise RuntimeError(f"Failed to spawn {failed} task managers")

            # Register the task manager configs and IDs for the app
            assert len(task_manager_configs) == len(task_manager_ids)
            self.app_task_manager_configs[app_id] = task_manager_configs
            self.app_task_manager_ids[app_id] = task_manager_ids

            logger.info(
                "Successfully reconciled new app %s with task managers %s",
                app_id,
                task_manager_ids,
            )
        except Exception as e:
            logger.exception("Failed to spawn task managers: %s", e)
            raise
        finally:
            if self.app_lock.locked():
                self.app_lock.release()

    async def reconcile_removed_app(self, app_id: str) -> None:
        """Reconcile removed app by shutting down task managers if needed."""
        logger.info("Reconciling removed app %s", app_id)

        # A task manager can be shared by multiple apps.
        # We shut down a task manager when the last app that uses it is unregistered.
        await self.app_lock.acquire()
        try:
            # Get and remove task manager configs for this app
            task_manager_configs = self.app_task_manager_configs.pop(app_id, None)
            task_manager_ids = self.app_task_manager_ids.pop(app_id, None)
            if task_manager_configs is None or task_manager_ids is None:
                logger.error("App %s not found in registered apps", app_id)
                raise ValueError(f"App {app_id} not found in registered apps")

            # Get configs still in use by other apps
            active_configs = set()
            for configs in self.app_task_manager_configs.values():
                active_configs.update(configs)

            # For each config and ID from the removed app
            coros = []
            shutdown_task_manager_ids = []
            for config, id in zip(task_manager_configs, task_manager_ids, strict=True):
                # Only shut down if no other app is using this config
                if config not in active_configs:
                    shutdown_task_manager_ids.append(id)
                    coros.append(self._shutdown_task_manager(id))
            logger.info("Shutting down task managers %s", shutdown_task_manager_ids)
            results = await asyncio.gather(*coros, return_exceptions=True)

            # Check for errors
            failed = 0
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    logger.error(
                        "Failed to shut down task manager %s: %s",
                        task_manager_ids[i],
                        result,
                    )
                    failed += 1
                else:
                    logger.info("Successfully shut down task manager %s", task_manager_ids[i])
            if failed:
                raise RuntimeError(f"Failed to shutdown {failed} task managers")

            logger.info(
                "Successfully reconciled removed app %s with task managers %s by shutting down %s",
                app_id,
                task_manager_ids,
                shutdown_task_manager_ids,
            )
        except Exception as e:
            logger.exception("Failed to reconcile removed app %s: %s", app_id, e)
            raise
        finally:
            if self.app_lock.locked():
                self.app_lock.release()

    async def healthcheck(self) -> tuple[bool, list[tuple[str, bool]]]:
        """Check the health of all task managers.

        We intentionally do not hold any locks while performing the health check,
        because we don't want to block other operations. It's fine even if a new
        task manager is missed or an error arises from a task manager that is being
        shut down.

        Returns:
            Tuple of overall_healthy and list of (task_manager_id, healthy)
        """
        logger.info("Performing health check of all task managers")

        task_manager_statuses: list[tuple[str, bool]] = []
        all_healthy = True

        task_manager_ids = []
        check_tasks = []
        for task_manager_id, stub in self.task_manager_stubs.items():
            task_manager_ids.append(task_manager_id)
            check_tasks.append(stub.Healthcheck(task_manager_pb2.HealthcheckRequest(), timeout=1.0))

        # Wait for all health checks to complete
        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        for task_manager_id, result in zip(task_manager_ids, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Health check failed for task manager %s: %s", task_manager_id, str(result))
                task_manager_statuses.append((task_manager_id, False))
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
        await self.api_client.close()
        for channel in self.task_manager_channels.values():
            await channel.close(grace=1.0)

    async def _spawn_task_manager(self, task_manager_config: TaskManagerConfig) -> str:
        """Spawn a new task manager for the given task and return its ID.

        Upon success, the task manager ID is returned. If anything goes wrong,
        side effects are cleaned up and an exception is raised.
        """
        logger.info("Spawning task manager for %s", task_manager_config)

        # Sanity check task manager type
        if task_manager_config.type.upper() not in task_manager_pb2.TaskManagerType.keys():  # noqa: SIM118
            raise ValueError(f"Unknown task manager type: {task_manager_config.type}")

        # Create a unique task manager ID
        while True:
            task_manager_id = task_manager_config.create_id()
            if task_manager_id not in self.task_manager_stubs:
                break

        # Acquire the task manager lock
        await self.task_manager_lock.acquire()

        try:
            # Allocate resource starter pack for the task manager
            resource = self.resource.allocate(num_gpus=2, owner=task_manager_id)
            self.task_manager_resources[task_manager_id] = resource

            # Create a new task manager pod and service
            pod_name = f"task-manager-{task_manager_id}"
            service_name = f"task-manager-{task_manager_id}"
            self.task_manager_pods[task_manager_id] = pod_name
            self.task_manager_services[task_manager_id] = service_name

            pod = kclient.V1Pod(
                metadata=kclient.V1ObjectMeta(
                    name=pod_name,
                    labels={
                        "app": "task-manager",
                        "task-manager-id": task_manager_id,
                    },
                ),
                spec=kclient.V1PodSpec(
                    containers=[
                        kclient.V1Container(
                            name="task-manager",
                            image=constants.CONTAINER_IMAGE_TASK_MANAGER,
                            image_pull_policy="Always",
                            ports=[kclient.V1ContainerPort(container_port=50051, name="grpc")],
                        )
                    ],
                    service_account_name="task-manager",
                ),
            )
            service = kclient.V1Service(
                metadata=kclient.V1ObjectMeta(
                    name=service_name,
                    labels={
                        "app": "task-manager",
                        "task-manager-id": task_manager_id,
                    },
                ),
                spec=kclient.V1ServiceSpec(
                    selector={
                        "app": "task-manager",
                        "task-manager-id": task_manager_id,
                    },
                    ports=[kclient.V1ServicePort(port=50051, target_port="grpc")],
                ),
            )
            await self.kube_core_client.create_namespaced_pod(
                namespace=constants.K8S_NAMESPACE,
                body=pod,
            )  # type: ignore
            await self.kube_core_client.create_namespaced_service(
                namespace=constants.K8S_NAMESPACE,
                body=service,
            )  # type: ignore
            logger.info("Created task manager pod %s and service %s", pod_name, service_name)

            # Connect to the task manager gRPC server to initialize it
            channel = grpc.aio.insecure_channel(f"{service_name}:50051")
            stub = task_manager_pb2_grpc.TaskManagerStub(channel)
            self.task_manager_channels[task_manager_id] = channel
            self.task_manager_stubs[task_manager_id] = stub

            register_task_req = task_manager_pb2.RegisterTaskRequest(
                task_manager_id=task_manager_id,
                type=getattr(task_manager_pb2.TaskManagerType, task_manager_config.type.upper()),
                gpus=[gpu.to_pb(add=True) for gpu in resource],
                config=task_manager_config.model_dump_json(),
            )

            response: task_manager_pb2.RegisterTaskResponse = await stub.RegisterTask(
                register_task_req, wait_for_ready=True
            )
            if response.status != common_pb2.Status.STATUS_OK:
                raise RuntimeError(f"Failed to register task manager: {response}")
        except Exception as e:
            logger.exception("Failed to spawn task manager: %s", e)
            await self._shutdown_task_manager(task_manager_id)
        finally:
            if self.task_manager_lock.locked():
                self.task_manager_lock.release()

        return task_manager_id

    async def _shutdown_task_manager(self, task_manager_id: str) -> None:
        """Shutdown the task manager and release its resources.

        This method is called in two cases:
        1. As a cleanup routine when the task manager fails to start.
        2. When the task manager is no longer needed and should be shut down.
            This is when the last app that uses the task manager is unregistered.

        In the first case, this method is called with the task manager lock held.
        In the second case, this method is called without the task manager lock held.

        Args:
            task_manager_id: The ID of the task manager to shut down.
        """
        logger.info("Shutting down task manager %s", task_manager_id)

        # Acquire the task manager lock
        entered_with_lock = self.task_manager_lock.locked()
        if not entered_with_lock:
            await self.task_manager_lock.acquire()

        try:
            # Shutdown the task manager gRPC server
            if stub := self.task_manager_stubs.pop(task_manager_id, None):
                shutdown_req = task_manager_pb2.ShutdownRequest()
                with suppress(grpc.aio.AioRpcError):
                    await stub.Shutdown(shutdown_req)
            if channel := self.task_manager_channels.pop(task_manager_id, None):
                await channel.close()

            # Release GPU resources allocated to the task manager
            if resources := self.task_manager_resources.pop(task_manager_id, None):
                for gpu in resources:
                    gpu.free()

            # Delete the task manager pod and service
            if pod_name := self.task_manager_pods.pop(task_manager_id, None):
                with suppress(kclient.ApiException):
                    await self.kube_core_client.delete_namespaced_pod(
                        name=pod_name,
                        namespace=constants.K8S_NAMESPACE,
                    )  # type: ignore
            if service_name := self.task_manager_services.pop(task_manager_id, None):
                with suppress(kclient.ApiException):
                    await self.kube_core_client.delete_namespaced_service(
                        name=service_name,
                        namespace=constants.K8S_NAMESPACE,
                    )  # type: ignore
        except Exception as e:
            logger.exception(
                "An unexpected exception aborted the shutdown of task manager %s: %s",
                task_manager_id,
                e,
            )
            raise
        finally:
            if not entered_with_lock and self.task_manager_lock.locked():
                self.task_manager_lock.release()
