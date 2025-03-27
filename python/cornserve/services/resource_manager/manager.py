"""Core resource manager class."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from typing import Coroutine

import grpc
import kubernetes_asyncio.client as kclient
import kubernetes_asyncio.config as kconfig
from opentelemetry import trace

from cornserve import constants
from cornserve.frontend.tasks import Task
from cornserve.logging import get_logger
from cornserve.services.pb import (
    common_pb2,
    task_dispatcher_pb2,
    task_dispatcher_pb2_grpc,
    task_manager_pb2,
    task_manager_pb2_grpc,
)
from cornserve.services.resource_manager.resource import GPU, Resource
from cornserve.services.task_manager.models import TaskManagerConfig

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class TaskManagerDeployment:
    """Informational data structure for a deployed task manager.

    Fields are populated by either freshly spawning a task manager or
    sharing an already-running task manager. After the new app has been
    deployed, `id` and `url` should be populated; if not, it's a bug.

    Note that task managers can be shared by multiple apps. Thus, some
    AppDeployment objects may be holding the exact same TaskManagerDeployment
    object (i.e., same config, ID, and URL).
    """

    config: TaskManagerConfig
    id: str = ""
    url: str = ""

    def __repr__(self) -> str:
        """Return a string representation of the task manager."""
        string = f"DeployedTaskManager(config={self.config}"
        if self.id:
            string += f", id={self.id}"
        if self.url:
            string += f", url={self.url}"
        string += ")"
        return string


@dataclass
class TaskDeployment:
    """Informational data structure for a deployed task."""

    task: Task
    task_managers: list[TaskManagerDeployment]

    def search_task_manager_config(self, config: TaskManagerConfig) -> TaskManagerDeployment | None:
        """Search for a task manager deployment with the given config."""
        for task_manager in self.task_managers:
            if task_manager.config == config:
                assert task_manager.id
                assert task_manager.url
                return task_manager
        return None


@dataclass
class AppDeployment:
    """Informational data structure for a deployed app."""

    id: str
    tasks: list[TaskDeployment]

    def search_task_manager_config(self, config: TaskManagerConfig) -> TaskManagerDeployment | None:
        """Search for a task manager deployment with the given config."""
        for task in self.tasks:
            if (task_manager := task.search_task_manager_config(config)) is not None:
                return task_manager
        return None


class ResourceManager:
    """The Resource Manager allocates resources for Task Managers."""

    def __init__(self, api_client: kclient.ApiClient, resource: Resource) -> None:
        """Initialize the ResourceManager."""
        self.api_client = api_client
        self.resource = resource

        self.kube_core_client = kclient.CoreV1Api(api_client)

        # Task dispatcher gRPC handles
        self.task_dispatcher_channel = grpc.aio.insecure_channel(constants.K8S_TASK_DISPATCHER_GRPC_URL)
        self.task_dispatcher_stub = task_dispatcher_pb2_grpc.TaskDispatcherStub(self.task_dispatcher_channel)

        # App state
        self.app_lock = asyncio.Lock()
        self.app_deployments: dict[str, AppDeployment] = {}

        # Task manager state
        self.task_manager_resources: dict[str, list[GPU]] = {}
        self.task_manager_pods: dict[str, str] = {}
        self.task_manager_services: dict[str, str] = {}
        self.task_manager_channels: dict[str, grpc.aio.Channel] = {}
        self.task_manager_stubs: dict[str, task_manager_pb2_grpc.TaskManagerStub] = {}

    @staticmethod
    async def init() -> ResourceManager:
        """Actually initialize the resource manager.

        First, discover all sidecar pods in the cluster and instantiate one GPU object
        for each pod. The global rank is parsed from the pod name, and the local rank is
        determined by the alphabetical order of pod names within each node. Created GPU
        objects make up the `Resource` object.

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
                name=constants.K8S_SIDECAR_STATEFULSET_NAME,
                namespace=constants.K8S_NAMESPACE,
            )
            num_ready = sidecar_set.status.ready_replicas or 0  # type: ignore
            num_expected = sidecar_set.spec.replicas  # type: ignore
            if num_ready == num_expected:
                break
            logger.info("Waiting for all sidecar pods to be ready... %d/%d", num_ready, num_expected)
        logger.info("All %d sidecar pods are ready.", sidecar_set.status.ready_replicas)  # type: ignore

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

        # Construct GPUs and cluster resource object
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

    @tracer.start_as_current_span("ResourceManager.reconcile_new_app")
    async def reconcile_new_app(self, app_id: str, tasks: list[Task]) -> None:
        """Reconcile new app by spawning task managers if needed."""
        logger.info("Reconcile new app %s with tasks %s", app_id, tasks)
        span = trace.get_current_span()
        span.set_attribute("resource_manager.reconcile_new_app.app_id", app_id)

        # Construct app deployment for the app
        # Fields of TaskManagerDeployments (id and url) will be populated by either
        # freshly spawning a task manager or sharing an already-running task manager.
        task_deployments: list[TaskDeployment] = []
        for task in tasks:
            task_manager_configs = TaskManagerConfig.from_task(task)
            task_manager_deployments = [
                TaskManagerDeployment(config=task_manager_config) for task_manager_config in task_manager_configs
            ]
            task_deployments.append(TaskDeployment(task=task, task_managers=task_manager_deployments))
        app_deployment = AppDeployment(id=app_id, tasks=task_deployments)
        logger.info("Task manager configs: %s", task_deployments)

        # A task manager can be shared by multiple apps.
        # We only spawn task managers that are not already running.
        await self.app_lock.acquire()
        try:
            # Check if the app is already registered
            if app_id in self.app_deployments:
                raise ValueError(f"App {app_id} already registered")

            # Determine which task managers to spawn and which ones to share
            task_manager_deployments_to_spawn: list[TaskManagerDeployment] = []
            coros: list[Coroutine[None, None, tuple[str, str]]] = []
            for task_deployment in app_deployment.tasks:
                for task_manager_deployment in task_deployment.task_managers:
                    # This task manager is already running as part of another app
                    for existing_app_deplyment in self.app_deployments.values():
                        matched_tm_deployment = existing_app_deplyment.search_task_manager_config(
                            task_manager_deployment.config
                        )
                        if matched_tm_deployment is not None:
                            task_manager_deployment.id = matched_tm_deployment.id
                            task_manager_deployment.url = matched_tm_deployment.url
                            break
                    # This particular task manager config is not running anywhere in the cluster.
                    # It should be freshly spawned.
                    else:
                        task_manager_deployments_to_spawn.append(task_manager_deployment)
                        coros.append(self._spawn_task_manager(task_manager_deployment.config))
            span.set_attribute(
                "resource_manager.reconcile_new_app.task_manager_deployments_to_spawn",
                len(task_manager_deployments_to_spawn),
            )
            logger.info("Spawning task managers: %s", task_manager_deployments_to_spawn)
            spawn_results = await asyncio.gather(*coros, return_exceptions=True)

            # Check for errors
            failed = 0
            for task_manager_deployment, spawn_result in zip(
                task_manager_deployments_to_spawn,
                spawn_results,
                strict=True,
            ):
                if isinstance(spawn_result, BaseException):
                    logger.error(
                        "Failed to spawn task manager %s: %s",
                        task_manager_deployment.config,
                        spawn_result,
                    )
                    failed += 1
                else:
                    task_manager_deployment.id, task_manager_deployment.url = spawn_result
                    logger.info(
                        "Successfully spawned task manager %s: %s",
                        task_manager_deployment.config,
                        spawn_result,
                    )

            if failed:
                # Rollback all the task managers that were successfully spawned
                logger.info("Rolling back all the task managers that were successfully spawned")
                shutdown_coros = []
                for spawn_result in spawn_results:
                    if not isinstance(spawn_result, BaseException):
                        shutdown_coros.append(self._shutdown_task_manager(spawn_result[0]))
                await asyncio.gather(*shutdown_coros, return_exceptions=True)
                raise RuntimeError(f"Failed to spawn {failed} task managers")

            # Notify the task dispatcher of the new app and task managers
            task_infos: list[task_dispatcher_pb2.TaskInfo] = []
            for task_deployment in app_deployment.tasks:
                task_manager_infos: list[task_dispatcher_pb2.TaskManagerInfo] = []
                for task_manager_deployment in task_deployment.task_managers:
                    assert task_manager_deployment.id
                    assert task_manager_deployment.url
                    task_manager_info = task_dispatcher_pb2.TaskManagerInfo(
                        task_manager_id=task_manager_deployment.id,
                        type=task_manager_deployment.config.type,
                        url=task_manager_deployment.url,
                    )
                    task_manager_infos.append(task_manager_info)
                task_info = task_dispatcher_pb2.TaskInfo(
                    task_id=task_deployment.task.id,
                    type=task_deployment.task.to_type(),
                    task_config=task_deployment.task.model_dump_json(),
                    task_manager_info=task_manager_infos,
                )
                task_infos.append(task_info)

            await self.task_dispatcher_stub.NotifyAppRegistration(
                task_dispatcher_pb2.NotifyAppRegistrationRequest(app_id=app_id, tasks=task_infos)
            )

            self.app_deployments[app_id] = app_deployment
            logger.info("Successfully reconciled new app %s: %s", app_id, app_deployment)

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
            # Check if the app is registered
            if app_id not in self.app_deployments:
                logger.error("App %s not found in registered apps", app_id)
                raise ValueError(f"App {app_id} not found in registered apps")

            # Notify the Task Dispatcher of the removed app
            await self.task_dispatcher_stub.NotifyAppUnregistration(
                task_dispatcher_pb2.NotifyAppUnregistrationRequest(app_id=app_id)
            )

            # Get and remove the app deployment for this app
            app_deployment = self.app_deployments.pop(app_id)

            # Remove task managers that are no longer needed
            coros: list[Coroutine[None, None, None]] = []
            shutdown_task_managers = []
            for task_deployment in app_deployment.tasks:
                for task_manager_deployment in task_deployment.task_managers:
                    config = task_manager_deployment.config
                    for existing_app_deployment in self.app_deployments.values():
                        # Another app is using this task manager
                        if existing_app_deployment.search_task_manager_config(config) is not None:
                            break
                    # This task manager is no longer needed
                    else:
                        shutdown_task_managers.append(task_manager_deployment.config)
                        coros.append(self._shutdown_task_manager(task_manager_deployment.id))

            logger.info("Shutting down task managers %s", shutdown_task_managers)
            results = await asyncio.gather(*coros, return_exceptions=True)

            # Check for errors
            failed = 0
            for task_manager_config, result in zip(shutdown_task_managers, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error(
                        "Failed to shut down task manager %s: %s",
                        task_manager_config,
                        result,
                    )
                    failed += 1
                else:
                    logger.info("Successfully shut down task manager %s", task_manager_config)

            if failed:
                raise RuntimeError(f"Failed to shutdown {failed} task managers")

            logger.info(
                "Successfully reconciled removed app %s by shutting down %d task managers",
                app_deployment,
                len(shutdown_task_managers),
            )
        except Exception as e:
            logger.exception("Failed to reconcile removed app %s: %s", app_id, e)
            raise
        finally:
            if self.app_lock.locked():
                self.app_lock.release()

            self.resource.print_resource_status()

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
        await self.task_dispatcher_channel.close()
        close_coros = []
        for channel in self.task_manager_channels.values():
            close_coros.append(channel.close(grace=1.0))
        await asyncio.gather(*close_coros)

    @tracer.start_as_current_span("ResourceManager._spawn_task_manager")
    async def _spawn_task_manager(self, task_manager_config: TaskManagerConfig) -> tuple[str, str]:
        """Spawn a new task manager.

        Upon success, the task manager ID and URL are returned. If anything goes wrong,
        side effects are cleaned up and an exception is raised.
        """
        logger.info("Spawning task manager for %s", task_manager_config)
        span = trace.get_current_span()

        # Sanity check task manager type
        if task_manager_config.type.upper() not in task_manager_pb2.TaskManagerType.keys():  # noqa: SIM118
            raise ValueError(f"Unknown task manager type: {task_manager_config.type}")

        # Create a unique task manager ID
        while True:
            task_manager_id = task_manager_config.create_id()
            if task_manager_id not in self.task_manager_stubs:
                break
        span.set_attribute("resource_manager._spawn_task_manager.task_manager_id", task_manager_id)

        try:
            # Allocate resource starter pack for the task manager
            resource = self.resource.allocate(num_gpus=2, owner=task_manager_id)
            self.task_manager_resources[task_manager_id] = resource
            span.set_attribute("resource_manager._spawn_task_manager.gpus_allocated", len(resource))

            # Create a new task manager pod and service
            pod_name = f"task-manager-{task_manager_id}".lower()
            service_name = f"task-manager-{task_manager_id}".lower()
            port = 50051
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
                            ports=[kclient.V1ContainerPort(container_port=port, name="grpc")],
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
                    ports=[kclient.V1ServicePort(port=port, target_port="grpc")],
                ),
            )
            span.add_event("create_pod.start")
            await self.kube_core_client.create_namespaced_pod(
                namespace=constants.K8S_NAMESPACE,
                body=pod,
            )  # type: ignore
            span.add_event("create_pod.done")
            span.add_event("create_service.start")
            await self.kube_core_client.create_namespaced_service(
                namespace=constants.K8S_NAMESPACE,
                body=service,
            )  # type: ignore
            span.add_event("create_service.done")
            logger.info("Created task manager pod %s and service %s", pod_name, service_name)

            # Connect to the task manager gRPC server to initialize it
            channel = grpc.aio.insecure_channel(f"{service_name}:{port}")
            stub = task_manager_pb2_grpc.TaskManagerStub(channel)
            self.task_manager_channels[task_manager_id] = channel
            self.task_manager_stubs[task_manager_id] = stub

            # Initialize the task manager by providing it with the task it will manage
            # and an initial set of GPU resources to work with.
            span.add_event("register_task.start")
            register_task_req = task_manager_pb2.RegisterTaskRequest(
                task_manager_id=task_manager_id,
                type=getattr(task_manager_pb2.TaskManagerType, task_manager_config.type.upper()),
                gpus=[
                    task_manager_pb2.GPUResource(
                        action=task_manager_pb2.ResourceAction.ADD,
                        node_id=gpu.node,
                        global_rank=gpu.global_rank,
                        local_rank=gpu.local_rank,
                    )
                    for gpu in resource
                ],
                config=task_manager_config.model_dump_json(),
            )
            response: task_manager_pb2.RegisterTaskResponse = await stub.RegisterTask(
                register_task_req, wait_for_ready=True
            )
            if response.status != common_pb2.Status.STATUS_OK:
                raise RuntimeError(f"Failed to register task manager: {response}")
            span.add_event("register_task.done")

        except Exception as e:
            logger.exception("Failed to spawn task manager: %s", e)
            await self._shutdown_task_manager(task_manager_id)
            raise

        return (task_manager_id, f"{service_name}:{port}")

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
