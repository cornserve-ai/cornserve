"""Runtime task and descriptor registry service.

This module provides a single integration point for all services to:
- discover task definitions and execution descriptors at runtime
- register loaded classes into in-process registries and sys.modules
- create and retrieve task instances by name

Kubernetes specifics (custom resources, watching, etc.) are fully encapsulated
inside this module and are not exposed through its public API.
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any

from kubernetes_asyncio import client, config
from kubernetes_asyncio.watch import Watch

from cornserve.constants import CRD_GROUP, CRD_VERSION, K8S_NAMESPACE
from cornserve.logging import get_logger
from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY

if TYPE_CHECKING:
    from cornserve.task.base import UnitTask


logger = get_logger(__name__)


class TaskRegistry:
    """Service responsible for runtime registry population and access.

    Public methods avoid leaking any underlying storage concepts.
    """

    def __init__(self) -> None:
        self._api_client: client.ApiClient | None = None
        self._custom_api: client.CustomObjectsApi | None = None

    async def _load_config(self) -> None:
        if self._api_client:
            return
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config.")
        except config.ConfigException as e:
            logger.error("Failed to load Kubernetes config: %s", e)
            raise RuntimeError("Could not load Kubernetes configuration") from e

        self._api_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(self._api_client)

    async def create_task_definition(
        self,
        name: str,
        task_class_name: str,
        module_name: str,
        source_code: str,
        is_unit_task: bool = True,
        namespace: str = K8S_NAMESPACE,
    ) -> dict[str, Any]:
        """Create a task definition from source code."""
        await self._load_config()
        assert self._custom_api is not None

        encoded_source = base64.b64encode(source_code.encode("utf-8")).decode("utf-8")

        body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "TaskDefinition",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {
                "taskClassName": task_class_name,
                "moduleName": module_name,
                "sourceCode": encoded_source,
                "isUnitTask": is_unit_task,
            },
        }

        try:
            return await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural="taskdefinitions",
                body=body,
            )
        except client.ApiException as e:
            if e.status == 409:
                raise ValueError(f"Task definition {name} already exists") from e
            raise

    async def create_task_instance_from_task(
        self,
        task: UnitTask,
        task_uuid: str
    ) -> str:
        """Create a named task instance from a configured task object.

        Returns the instance name.
        """
        await self._load_config()
        assert self._custom_api is not None

        task_type = task.__class__.__name__.lower()
        instance_name = f"{task_type}-{task_uuid}"

        body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "UnitTaskInstance",
            "metadata": {"name": instance_name, "namespace": K8S_NAMESPACE},
            "spec": {
                "definitionRef": task.__class__.__name__,
                "config": task.model_dump(mode="json"),
                "executionDescriptorName": task.execution_descriptor_name,
            },
        }

        try:
            await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural="unittaskinstances",
                body=body,
            )
            logger.info("Created task instance: %s", instance_name)
            return instance_name
        except client.ApiException as e:
            if e.status == 409:
                raise ValueError(f"Task instance {instance_name} already exists") from e
            raise

    async def create_execution_descriptor(
        self,
        name: str,
        task_class_name: str,
        descriptor_class_name: str,
        module_name: str,
        source_code: str,
        is_default: bool = True,
    ) -> dict[str, Any]:
        """Create an execution descriptor from source code."""
        await self._load_config()
        assert self._custom_api is not None

        encoded_source = base64.b64encode(source_code.encode("utf-8")).decode("utf-8")

        body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "ExecutionDescriptor",
            "metadata": {"name": name, "namespace": K8S_NAMESPACE},
            "spec": {
                "taskClassName": task_class_name,
                "descriptorClassName": descriptor_class_name,
                "moduleName": module_name,
                "sourceCode": encoded_source,
                "isDefault": is_default,
            },
        }

        try:
            return await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural="executiondescriptors",
                body=body,
            )
        except client.ApiException as e:
            if e.status == 409:
                raise ValueError(f"Execution descriptor {name} already exists") from e
            raise

    async def get_task_instance(
        self, instance_name: str
    ) -> UnitTask:
        """Reconstruct a configured task from its instance name."""
        await self._load_config()
        assert self._custom_api is not None

        try:
            cr_object = await self._custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural="unittaskinstances",
                name=instance_name,
            )

            spec = cr_object.get("spec", {})
            definition_ref = spec.get("definitionRef")
            config = spec.get("config")

            if not definition_ref:
                raise ValueError(f"Task instance {instance_name} missing definitionRef")
            if not config:
                raise ValueError(f"Task instance {instance_name} missing config")

            task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task(definition_ref)
            task_instance = task_cls.model_validate(config)
            logger.info("Reconstructed %s from task instance: %s", definition_ref, instance_name)
            return task_instance
        except client.ApiException as e:
            if e.status == 404:
                raise ValueError(f"Task instance {instance_name} not found") from e
            raise RuntimeError(f"Failed to get task instance {instance_name}: {e}") from e

    async def watch_updates(self) -> None:
        """Background task to populate registries by watching definitions and descriptors."""
        await self._load_config()
        watchers = [
            self._watch_taskdefinitions(),
            self._watch_executiondescriptors(),
        ]
        await asyncio.gather(*watchers)

    def _handle_object(self, obj: dict[str, Any], kind: str, event_type: str) -> None:
        spec = obj.get("spec", {})
        metadata = obj.get("metadata", {})
        name = metadata.get("name")

        if kind == "TaskDefinition" and event_type in ("EXISTING", "ADDED", "MODIFIED"):
            try:
                task_class_name = spec.get("taskClassName")
                module_name = spec.get("moduleName")
                source_code = spec.get("sourceCode")
                is_unit_task = spec.get("isUnitTask")

                if not task_class_name or not module_name or not source_code or is_unit_task is None:
                    logger.error("Task definition %s missing required fields", name)
                    return

                TASK_CLASS_REGISTRY.load_from_source(
                    source_code=source_code,
                    task_class_name=task_class_name,
                    module_name=module_name,
                    is_unit_task=is_unit_task,
                )
                logger.info("Registered %s task: %s", "unit" if is_unit_task else "composite", task_class_name)
                
                # Only bind descriptors for unit tasks
                if is_unit_task:
                    try:
                        task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task(task_class_name)
                        DESCRIPTOR_REGISTRY.bind_pending_descriptor_for_task_class(task_cls)
                    except Exception:
                        # If task not fully available, binding will happen on later attempts
                        pass
            except Exception as e:
                logger.error("Failed to register task %s from %s: %s", task_class_name if 'task_class_name' in locals() else 'unknown', name, e)

        elif kind == "ExecutionDescriptor" and event_type in ("EXISTING", "ADDED", "MODIFIED"):
            try:
                descriptor_class_name = spec.get("descriptorClassName")
                module_name = spec.get("moduleName")
                source_code = spec.get("sourceCode")
                task_class_name = spec.get("taskClassName")

                if not descriptor_class_name or not module_name or not source_code or not task_class_name:
                    logger.error("Execution descriptor %s missing required fields", name)
                    return

                DESCRIPTOR_REGISTRY.load_from_source(
                    source_code=source_code,
                    descriptor_class_name=descriptor_class_name,
                    module_name=module_name,
                    task_class_name=task_class_name,
                )
                logger.info("Registered execution descriptor: %s", descriptor_class_name)
            except Exception as e:
                logger.error(
                    "Failed to register execution descriptor %s from %s: %s",
                    descriptor_class_name if 'descriptor_class_name' in locals() else 'unknown',
                    name,
                    e,
                )

    async def _watch_resource(self, plural: str, kind: str) -> None:
        assert self._custom_api is not None

        # Save the last resourceVersion seen to resume watch without relisting
        resource_version: str | None = None

        while True:
            try:
                # Only relist when we don't have a resourceVersion
                if resource_version is None:
                    initial_list = await self._custom_api.list_namespaced_custom_object(
                        group=CRD_GROUP,
                        version=CRD_VERSION,
                        namespace=K8S_NAMESPACE,
                        plural=plural,
                    )

                    for item in initial_list.get("items", []):
                        self._handle_object(item, kind, "EXISTING")

                    resource_version = initial_list.get("metadata", {}).get("resourceVersion")

                w = Watch()
                async with w.stream(
                    self._custom_api.list_namespaced_custom_object,
                    group=CRD_GROUP,
                    version=CRD_VERSION,
                    namespace=K8S_NAMESPACE,
                    plural=plural,
                    watch=True,
                    resource_version=resource_version,
                    timeout_seconds=300,
                ) as stream:
                    async for event in stream:
                        event_type = event.get("type", "UNKNOWN")
                        obj = event["object"]
                        self._handle_object(obj, kind, event_type)
                        # Update resourceVersion
                        try:
                            rv = obj.get("metadata", {}).get("resourceVersion")
                            if rv:
                                resource_version = rv
                        except Exception:
                            logger.error("Failed to get resourceVersion for %s", kind)
            except asyncio.CancelledError:
                raise
            except client.ApiException as e:
                # If the resourceVersion is too old, the API returns 410 Gone.
                # Reset resourceVersion to force a relist on next loop.
                if getattr(e, "status", None) == 410:
                    logger.warning("Watch for %s expired (410 Gone). Relisting to refresh.", kind)
                    resource_version = None
                    continue
                logger.error("Error watching %s (API): %s", kind, e)
                await asyncio.sleep(5)
            except Exception as e:
                logger.error("Error watching %s: %s", kind, e)
                await asyncio.sleep(5)

    async def _watch_taskdefinitions(self) -> None:
        await self._watch_resource("taskdefinitions", "TaskDefinition")

    async def _watch_executiondescriptors(self) -> None:
        await self._watch_resource("executiondescriptors", "ExecutionDescriptor")

    async def shutdown(self) -> None:
        if self._api_client:
            await self._api_client.close()
            self._api_client = None
            self._custom_api = None


