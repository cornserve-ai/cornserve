# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportPossiblyUnboundVariable=false, reportArgumentType=false
"""Utilities used by services for task/descriptor discovery and registry population.

Currently, this module provides utilities for interacting with k8s CRDs to:
- discover task definitions and execution descriptors at runtime
- populate in-process registries and ``sys.modules``
- create and retrieve unit task instances

NOTE: we disable pyright warnings in this file because the k8s CustomObjectsApi
return types are defined by the CRD schemas, which pyright does not understand.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kubernetes_asyncio import client, config
from kubernetes_asyncio.watch import Watch

from cornserve.constants import (
    CR_NAME_LATEST_TASKLIB_RV,
    CRD_GROUP,
    CRD_KIND_EXECUTION_DESCRIPTOR,
    CRD_KIND_TASK_DEFINITION,
    CRD_KIND_UNIT_TASK_INSTANCE,
    CRD_PLURAL_EXECUTION_DESCRIPTORS,
    CRD_PLURAL_LATEST_TASKLIB_RVS,
    CRD_PLURAL_TASK_DEFINITIONS,
    CRD_PLURAL_UNIT_TASK_INSTANCES,
    CRD_VERSION,
    K8S_NAMESPACE,
    TASKLIB_DIR,
)
from cornserve.logging import get_logger
from cornserve.services.task_registry.descriptor_registry import DESCRIPTOR_REGISTRY
from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY
from cornserve.services.task_registry.utils import purge_tasklib_modules_and_delete_dir

if TYPE_CHECKING:
    from cornserve.task.base import UnitTask


logger = get_logger(__name__)


class TaskRegistry:
    """Utilities for interacting with k8s CRs for task related classes and instances."""

    def __init__(self) -> None:
        """Initialize lazy Kubernetes API clients."""
        self._api_client: client.ApiClient | None = None
        self._custom_api: client.CustomObjectsApi | None = None

        # Watcher resource version states (updated by background watch tasks)
        self._task_definition_rv: int = 0
        self._execution_descriptor_rv: int = 0

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

    async def update_latest_tasklib_rv(
        self,
        max_task_class_rv: int,
        max_descriptor_rv: int,
        *,
        namespace: str = K8S_NAMESPACE,
        name: str = CR_NAME_LATEST_TASKLIB_RV,
    ) -> None:
        """Update the singleton LatestTasklibRV CR instance with the latest tasklib RVs.

        NOTE: This CR is expected to always exist (initialized at cluster creation time).
        """
        await self._load_config()
        assert self._custom_api is not None

        max_task_class_rv = int(max_task_class_rv)
        max_descriptor_rv = int(max_descriptor_rv)
        patch_body = [
            {"op": "replace", "path": "/spec/maxTaskClassRV", "value": max_task_class_rv},
            {"op": "replace", "path": "/spec/maxDescriptorRV", "value": max_descriptor_rv},
        ]

        await self._custom_api.patch_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural=CRD_PLURAL_LATEST_TASKLIB_RVS,
            name=name,
            body=patch_body,
        )

    async def get_latest_tasklib_rv(
        self,
        namespace: str = K8S_NAMESPACE,
        name: str = CR_NAME_LATEST_TASKLIB_RV,
    ) -> tuple[int, int]:
        """Read the latest tasklib resource versions from the LatestTasklibRV CR."""
        await self._load_config()
        assert self._custom_api is not None

        try:
            cr = await self._custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural=CRD_PLURAL_LATEST_TASKLIB_RVS,
                name=name,
            )
            spec = cr.get("spec", {})
            max_task_class_rv = int(spec.get("maxTaskClassRV", 0))
            max_descriptor_rv = int(spec.get("maxDescriptorRV", 0))
            return (max_task_class_rv, max_descriptor_rv)
        except Exception as e:
            logger.warning("Failed to read LatestTasklibRV CR: %s", e)
            return (0, 0)

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
            "kind": CRD_KIND_TASK_DEFINITION,
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
                plural=CRD_PLURAL_TASK_DEFINITIONS,
                body=body,
            )
        except client.ApiException as e:
            if e.status == 409:
                # TODO: think about versioning here. If we have a mechanism to invalidate the
                # the loaded task definitions, we can allow user to update the CR rather than error out.
                raise ValueError(f"Task definition {name} already exists") from e
            raise

    async def create_task_instance_from_task(self, task: UnitTask, task_uuid: str) -> str:
        """Create a named task instance from a configured task object.

        Returns the instance name.
        """
        await self._load_config()
        assert self._custom_api is not None

        task_type = task.__class__.__name__.lower()
        instance_name = f"{task_type}-{task_uuid}"

        body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": CRD_KIND_UNIT_TASK_INSTANCE,
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
                plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
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
            "kind": CRD_KIND_EXECUTION_DESCRIPTOR,
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
                plural=CRD_PLURAL_EXECUTION_DESCRIPTORS,
                body=body,
            )
        except client.ApiException as e:
            if e.status == 409:
                raise ValueError(f"Execution descriptor {name} already exists") from e
            raise

    async def check_emptiness(self) -> bool:
        """Return True if there are no task definitions deployed in the cluster."""
        await self._load_config()
        assert self._custom_api is not None

        try:
            resp = await self._custom_api.list_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_TASK_DEFINITIONS,
                limit=1,
            )
            items = resp.get("items", [])
            return len(items) == 0
        except client.ApiException as e:
            logger.error("Failed to list task definitions for emptiness check: %s", e)
            raise RuntimeError(f"Failed to check task definitions: {e}") from e

    async def check_no_active_task_instance(self) -> bool:
        """Return True if there are NO UnitTaskInstance CRs present (i.e. cluster idle)."""
        await self._load_config()
        assert self._custom_api is not None
        try:
            resp = await self._custom_api.list_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
                limit=1,
            )
            items = resp.get("items", [])
            return len(items) == 0
        except client.ApiException as e:
            logger.error("Failed to list unit task instances: %s", e)
            raise RuntimeError(f"Failed to check unit task instances: {e}") from e

    async def delete_all_task_definitions_and_descriptors(self) -> None:
        """Delete all task class definition and execution descriptor CRs."""
        await self._load_config()
        assert self._custom_api is not None

        await self._delete_all_crs_by_plural(CRD_PLURAL_EXECUTION_DESCRIPTORS)
        await self._delete_all_crs_by_plural(CRD_PLURAL_TASK_DEFINITIONS)

        logger.info("Deleted all task definitions and execution descriptors.")

    async def get_task_instance(self, instance_name: str) -> UnitTask:
        """Reconstruct a configured task from its instance name."""
        await self._load_config()
        assert self._custom_api is not None

        try:
            cr_object = await self._custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
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

    async def delete_task_instance(self, instance_name: str) -> None:
        """Delete a unit task instance CR by instancename."""
        await self._load_config()
        assert self._custom_api is not None

        try:
            await self._custom_api.delete_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
                name=instance_name,
            )
            logger.info("Deleted task instance CR: %s", instance_name)
        except client.ApiException as e:
            # 404 means it's already gone; treat as success is okay
            if getattr(e, "status", None) == 404:
                logger.info("Task instance CR to be deleted is already absent: %s", instance_name)
                return
            logger.error("Failed to delete task instance %s: %s", instance_name, e)
            raise RuntimeError(f"Failed to delete task instance {instance_name}: {e}") from e

    async def watch_updates(self) -> None:
        """Background task to populate registries by watching definitions and descriptors."""
        await self._load_config()
        watchers = [
            self._watch_taskdefinitions(),
            self._watch_executiondescriptors(),
        ]
        await asyncio.gather(*watchers)

    async def sync_watchers(self, poll_interval: float = 0.1) -> None:
        """Wait until both watchers have caught up to their respective target RVs."""
        target_task_class_rv, target_descriptor_rv = await self.get_latest_tasklib_rv()

        while self._task_definition_rv < target_task_class_rv or self._execution_descriptor_rv < target_descriptor_rv:
            await asyncio.sleep(poll_interval)

    async def _delete_all_crs_by_plural(self, plural: str) -> None:
        try:
            resp = await self._custom_api.list_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=plural,
            )
            items = resp.get("items", [])

            async def _delete_one(name: str) -> None:
                try:
                    await self._custom_api.delete_namespaced_custom_object(
                        group=CRD_GROUP,
                        version=CRD_VERSION,
                        namespace=K8S_NAMESPACE,
                        plural=plural,
                        name=name,
                    )
                except client.ApiException as e:
                    # Ignore not found
                    if e.status != 404:
                        logger.error("Failed deleting %s/%s: %s", plural, name, e)
                        raise

            coroutines = []
            for item in items:
                name = item.get("metadata", {}).get("name")
                coroutines.append(_delete_one(name))

            await asyncio.gather(*coroutines)
        except client.ApiException as e:
            logger.error("Failed to list %s for purge: %s", plural, e)
            raise

    def _purge_local_registries_if_needed(self) -> None:
        """Purge in-process runtime state and the source files in tasklib directory.

        To avoid repeated cleanup on consecutive DELETE events, we check whether
        the tasklib directory exists. If absent, skip.
        """
        if not Path(TASKLIB_DIR).exists():
            # Already absent
            return

        # Clear registries
        try:
            DESCRIPTOR_REGISTRY.clear()
            TASK_CLASS_REGISTRY.clear()
            purge_tasklib_modules_and_delete_dir()
        except Exception as e:
            logger.warning("Error clearing registries and purging tasklib: %s", e)
            raise

    def _handle_object(self, obj: dict[str, Any], kind: str, event_type: str) -> None:
        spec = obj.get("spec", {})
        metadata = obj.get("metadata", {})
        name = metadata.get("name")

        if kind == CRD_KIND_TASK_DEFINITION and event_type in ("EXISTING", "ADDED", "MODIFIED"):
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

                # Only bind descriptors for unit tasks
                if is_unit_task:
                    try:
                        task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task(task_class_name)
                        DESCRIPTOR_REGISTRY.bind_pending_descriptor_for_task_class(task_cls)
                    except Exception:
                        # If task not fully available, binding will happen on later attempts
                        pass
            except Exception as e:
                logger.error(
                    "Failed to register task %s from %s: %s",
                    task_class_name if "task_class_name" in locals() else "unknown",
                    name,
                    e,
                )

        elif kind == CRD_KIND_EXECUTION_DESCRIPTOR and event_type in ("EXISTING", "ADDED", "MODIFIED"):
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
            except Exception as e:
                logger.error(
                    "Failed to register execution descriptor %s from %s: %s",
                    descriptor_class_name if "descriptor_class_name" in locals() else "unknown",
                    name,
                    e,
                )

        # NOTE: We don't allow individual deletion for now. Any DELETED event triggers a complete purge.
        elif event_type == "DELETED" and kind in (CRD_KIND_TASK_DEFINITION, CRD_KIND_EXECUTION_DESCRIPTOR):
            try:
                self._purge_local_registries_if_needed()
            except Exception as e:
                logger.error("Error occurred during purge: %s", e)
                raise

    def _update_watcher_rv(self, kind: str, rv: int) -> None:
        """Update the appropriate watcher's resource version state."""
        if kind == CRD_KIND_TASK_DEFINITION:
            self._task_definition_rv = rv
        elif kind == CRD_KIND_EXECUTION_DESCRIPTOR:
            self._execution_descriptor_rv = rv

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
                    if resource_version:
                        self._update_watcher_rv(kind, int(resource_version))

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
                        rv = obj.get("metadata", {}).get("resourceVersion")
                        if rv:
                            resource_version = rv
                            self._update_watcher_rv(kind, int(rv))
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
        await self._watch_resource(CRD_PLURAL_TASK_DEFINITIONS, CRD_KIND_TASK_DEFINITION)

    async def _watch_executiondescriptors(self) -> None:
        await self._watch_resource(CRD_PLURAL_EXECUTION_DESCRIPTORS, CRD_KIND_EXECUTION_DESCRIPTOR)

    async def shutdown(self) -> None:
        """Close underlying Kubernetes client resources."""
        if self._api_client:
            await self._api_client.close()
            self._api_client = None
            self._custom_api = None
