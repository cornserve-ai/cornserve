"""Core resource manager class."""

from __future__ import annotations

import asyncio
from collections import defaultdict

import kubernetes_asyncio.config as kconfig
import kubernetes_asyncio.client as kclient

from cornserve.constants import K8S_NAMESPACE
from cornserve.logging import get_logger

from .resource import GPU, Resource

logger = get_logger(__name__)


class ResourceManager:
    """The Resource Manager allocates resources for Task Managers."""

    def __init__(self, api_client: kclient.ApiClient, resource: Resource) -> None:
        """Initialize the ResourceManager."""
        self.api_client = api_client
        self.resource = resource

    @staticmethod
    async def init() -> ResourceManager:
        """Initialize the resource manager."""
        kconfig.load_incluster_config()
        api_client = kclient.ApiClient()
        core_api = kclient.CoreV1Api(api_client)
        apps_api = kclient.AppsV1Api(api_client)

        # Wait until the sidecars are all ready
        while True:
            await asyncio.sleep(1)
            sidecar_set = await apps_api.read_namespaced_stateful_set(name="sidecar", namespace=K8S_NAMESPACE)
            if sidecar_set.status.ready_replicas == sidecar_set.spec.replicas:
                break
            logger.info("Waiting for sidecar pods to be ready...")
        logger.info("All sidecar %d pods are ready.", sidecar_set.status.ready_replicas)

        # Discover the sidecar pods
        label_selector = ",".join(f"{key}={value}" for key, value in sidecar_set.spec.selector.match_labels.items())
        sidecar_pod_list = await core_api.list_namespaced_pod(namespace=K8S_NAMESPACE, label_selector=label_selector)
        sidecar_pods = sidecar_pod_list.items

        # Construct cluster resource object
        node_to_pods: dict[str, list[kclient.V1Pod]] = defaultdict(list)
        for pod in sidecar_pods:
            node = pod.spec.node_name
            node_to_pods[node].append(pod)
        gpus = []
        for node, pods in node_to_pods.items():
            for local_rank, pod in enumerate(sorted(pods, key=lambda p: p.metadata.name)):
                global_rank = int(pod.metadata.name.split("-")[-1])
                gpu = GPU(
                    id=global_rank,
                    node=node,
                    global_rank=global_rank,
                    local_rank=local_rank,
                )
                gpus.append(gpu)
        resource = Resource(gpus=gpus)

        return ResourceManager(api_client=api_client, resource=resource)

    async def shutdown(self) -> None:
        """Shutdown the ResourceManager."""
        await self.api_client.close()
