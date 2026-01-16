"""Utilities for CornServe services."""

from __future__ import annotations

import os
from datetime import datetime

import kubernetes_asyncio.client as kclient

from cornserve import constants
from cornserve.logging import get_logger

logger = get_logger(__name__)


async def discover_task_dispatcher_replicas(kube_client: kclient.CoreV1Api) -> list[str]:
    """Discover all Task Dispatcher replica endpoints via headless service.

    Uses Kubernetes service discovery to find all Task Dispatcher pod IPs
    and return their gRPC endpoints for broadcasting notifications.

    Args:
        kube_client: Kubernetes API client for service discovery

    Returns:
        List of Task Dispatcher gRPC URLs (e.g., ["10.1.2.3:50051", "10.1.2.4:50051"])

    Raises:
        RuntimeError: If Task Dispatcher replicas cannot be discovered.
    """
    try:
        # Query the headless service to get all Task Dispatcher pod endpoints
        endpoints = await kube_client.list_namespaced_endpoints(
            namespace=constants.K8S_NAMESPACE,
            field_selector=f"metadata.name={constants.K8S_TASK_DISPATCHER_HEADLESS_SERVICE}",
        )

        task_dispatcher_urls = []
        for endpoint in endpoints.items:
            if endpoint.subsets:
                for subset in endpoint.subsets:
                    if subset.addresses and subset.ports:
                        for address in subset.addresses:
                            for port in subset.ports:
                                if port.name == "grpc":
                                    task_dispatcher_urls.append(f"{address.ip}:{port.port}")

        if not task_dispatcher_urls:
            raise RuntimeError(
                f"No Task Dispatcher replicas found in headless service "
                f"{constants.K8S_TASK_DISPATCHER_HEADLESS_SERVICE}. "
                "Ensure Task Dispatcher pods are running and healthy."
            )

        logger.info("Discovered %d Task Dispatcher replicas: %s", len(task_dispatcher_urls), task_dispatcher_urls)
        return task_dispatcher_urls

    except Exception as e:
        raise RuntimeError(f"Failed to discover Task Dispatcher replicas: {e}") from e


def to_strict_k8s_name(name: str) -> str:
    """Normalize a name to be suitable even for the strictest Kubernetes requirements.

    RFC 1035 Label Names are the most restrictive:
    - contain at most 63 characters
    - contain only lowercase alphanumeric characters or '-'
    - start with an alphabetic character
    - end with an alphanumeric character

    Ref: https://kubernetes.io/docs/concepts/overview/working-with-objects/names
    """
    # Only lowercase alphanumeric characters and '-'
    name = name.lower()
    name = "".join(c if c.isalnum() or c == "-" else "-" for c in name)

    # Ensure length
    name = name[:63]

    # Starts and ends with an alphanumeric character
    name = name.strip("-")

    # Ensure it starts with an alphabetic character
    while name and name[0].isnumeric():
        name = name[1:]

    if not name:
        raise ValueError("Name cannot be empty after normalization.")

    return name


async def save_pod_logs(kube_client: kclient.CoreV1Api, pod_name: str) -> None:
    """Fetch and save the logs of a given pod to a file."""
    try:
        logs = await kube_client.read_namespaced_pod_log(
            name=pod_name,
            namespace=constants.K8S_NAMESPACE,
        )

        # Save to file
        log_dir = constants.K8S_LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_file = f"{log_dir}/{pod_name}_{timestamp}.log"

        with open(log_file, "w") as f:
            f.write(logs)
        logger.info("Pod logs saved to %s", log_file)
    except Exception as e:
        logger.exception("Failed to save pod logs for %s: %s", pod_name, e)
