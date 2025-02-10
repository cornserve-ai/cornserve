import asyncio

import grpc
import tyro
import kubernetes_asyncio.config as kconfig
import kubernetes_asyncio.client as kclient

from cornserve.logging import get_logger
from cornserve.services.pb import (
    resource_manager_pb2,
    resource_manager_pb2_grpc,
    task_manager_pb2,
    task_manager_pb2_grpc,
    common_pb2,
)

logger = get_logger(__name__)
cleanup_coroutines = []


class ResourceManagerServicer(resource_manager_pb2_grpc.ResourceManagerServicer):
    """Resource Manager gRPC service implementation."""

    def __init__(self, workers: dict[str, str]) -> None:
        """Initialize the ResourceManagerServicer.

        Args:
            workers: A dictionary of worker IDs and gRPC addresses.
        """
        self.workers = workers

        self.task_manager_to_workers = {}
        self.task_manager_stubs = {}

        kconfig.load_incluster_config()
        self.kube_client = kclient.CoreV1Api()

    async def DeployTaskManager(
        self,
        request: resource_manager_pb2.DeployTaskManagerRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.DeployTaskManagerResponse:
        """Deploy a task manager on a worker.

        This request comes from the Application Manager, which only knows *what* to run.
        """
        task_manager_id = request.task_manager_id
        if task_manager_id in self.task_manager_to_workers:
            await context.abort(
                grpc.StatusCode.ALREADY_EXISTS, "Task manager already exists"
            )

        # Create a new task manager pod and service
        pod_name = f"task-manager-{task_manager_id}"
        service_name = f"task-manager-{task_manager_id}"
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
                        image="cornserve/task-manager:latest",
                        ports=[
                            kclient.V1ContainerPort(container_port=50051, name="grpc")
                        ],
                        image_pull_policy="Always",
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
        await self.kube_client.create_namespaced_pod(namespace="cornserve", body=pod)
        await self.kube_client.create_namespaced_service(
            namespace="cornserve", body=service
        )
        logger.info(
            "Created task manager pod %s and service %s", pod_name, service_name
        )

        # Wait for the pod and service to be ready
        # while True:
        #     pod_ = await self.kube_client.read_namespaced_pod_status(namespace="cornserve", name=pod_name)
        #     service_ = await self.kube_client.read_namespaced_service(namespace="cornserve", name=service_name)
        #     if pod_.status.phase == "Running" and service_.spec.cluster_ip is not None:
        #         break
        #     await asyncio.sleep(1)

        # Connect to the task manager gRPC server to initialize it
        register_task_req = task_manager_pb2.RegisterTaskRequest()
        register_task_req.task_manager_id = task_manager_id
        register_task_req.descriptor.type = request.type
        # register_task_req.descriptor.config.MergeFrom(request.config)
        num_workers = 1
        for _ in range(num_workers):
            worker_id, worker_address = self.workers.popitem()
            register_task_req.workers[worker_id] = worker_address

        channel = grpc.aio.insecure_channel(f"{service_name}:50051")
        stub = task_manager_pb2_grpc.TaskManagerStub(channel)
        response: task_manager_pb2.RegisterTaskResponse = await stub.RegisterNewTask(
            register_task_req, wait_for_ready=True
        )
        if response.status != common_pb2.Status.STATUS_OK:
            await context.abort(
                grpc.StatusCode.INTERNAL, "Task manager registration failed"
            )

        # Update internal state
        self.task_manager_stubs[task_manager_id] = stub
        for worker_id, worker_address in register_task_req.workers.items():
            self.task_manager_to_workers[task_manager_id] = worker_id

        return resource_manager_pb2.DeployTaskManagerResponse(
            status=common_pb2.Status.STATUS_OK
        )

    async def Healthcheck(
        self,
        request: resource_manager_pb2.HealthcheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.HealthcheckResponse:
        """Recursively check and report the health of all workers."""
        resp = resource_manager_pb2.HealthcheckResponse()
        resp.status = common_pb2.Status.STATUS_OK
        for task_manager_id, task_manager_stub in self.task_manager_stubs.items():
            task_manager_resp: task_manager_pb2.HealthcheckResponse = (
                task_manager_stub.Healthcheck(task_manager_pb2.HealthcheckRequest())
            )
            worker_status = resource_manager_pb2.TaskManagerStatus(
                status=task_manager_resp.status
            )
            resp.task_manager_statuses[task_manager_id] = worker_status
        return resp


async def serve(ip: str = "[::]", port: int = 50051) -> None:
    server = grpc.aio.server()
    resource_manager_pb2_grpc.add_ResourceManagerServicer_to_server(
        ResourceManagerServicer({"worker-0": "worker-0:50051"}), server
    )
    listen_addr = f"{ip}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)

    await server.start()

    async def server_graceful_shutdown():
        logger.info("Starting graceful shutdown...")
        # Shuts down the server with 5 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(5)
        logger.info("Server stopped")

    cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(tyro.cli(serve))
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(asyncio.gather(*cleanup_coroutines))
        loop.close()
