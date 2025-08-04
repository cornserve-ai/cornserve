"""Manages Custom Resources (CRs) for Cornserve."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import yaml
from kubernetes_asyncio import client, config
from kubernetes_asyncio.watch import Watch

from cornserve.constants import CRD_GROUP, CRD_VERSION, K8S_NAMESPACE
from cornserve.logging import get_logger
from cornserve.task.registry import TASK_REGISTRY

logger = get_logger(__name__)


class CRManager:
    """A utility class for interacting with Cornserve Custom Resources."""

    def __init__(self) -> None:
        """Initialize the CRManager."""
        self._api_client: client.ApiClient | None = None
        self._custom_api: client.CustomObjectsApi | None = None

    async def _load_config(self) -> None:
        """Load Kubernetes configuration."""
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

    async def create_unit_task_definition(
        self, 
        name: str, 
        task_class_name: str,
        module_name: str,
        source_code: str,
        namespace: str = K8S_NAMESPACE
    ) -> dict[str, Any]:
        """Create a UnitTaskDefinition custom resource.
        
        Args:
            name: Name of the UnitTaskDefinition resource
            task_class_name: Name of the main UnitTask class in the source code
            module_name: Python module name where the task will be available
            source_code: Python source code for the task (will be base64 encoded)
            namespace: Kubernetes namespace to create the resource in
            
        Returns:
            The created custom resource object
        """
        await self._load_config()
        assert self._custom_api is not None

        # Base64 encode the source code
        encoded_source = base64.b64encode(source_code.encode('utf-8')).decode('utf-8')
        
        cr_body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "UnitTaskDefinition", 
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "taskClassName": task_class_name,
                "moduleName": module_name,
                "sourceCode": encoded_source
            }
        }

        try:
            result = await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural="unittaskdefinitions",
                body=cr_body
            )
            logger.info("Created UnitTaskDefinition: %s", name)
            return result
        except client.ApiException as e:
            if e.status == 409:
                logger.info("UnitTaskDefinition %s already exists", name)
                raise ValueError(f"UnitTaskDefinition {name} already exists") from e
            raise

    async def create_unit_task_instance_from_task(
        self,
        task: Any,  # UnitTask object - using Any to avoid circular import
        namespace: str = K8S_NAMESPACE
    ) -> tuple[dict[str, Any], str]:
        """Create a UnitTaskInstance custom resource from a UnitTask object.
        
        
        Args:
            task: The UnitTask object to serialize into a CR
            namespace: Kubernetes namespace to create the resource in
            
        Returns:
            Tuple of (created CR object, CR name) 
        """
        await self._load_config()
        assert self._custom_api is not None

        # Generate CR name: resource type + UUID (e.g., "encodertask-abc123")
        import uuid
        task_type = task.__class__.__name__.lower()
        unique_id = uuid.uuid4().hex[:8]  # Short UUID for readability
        cr_name = f"{task_type}-{unique_id}"

        # Create CR body with correct schema fields
        cr_body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "UnitTaskInstance",
            "metadata": {
                "name": cr_name,
                "namespace": namespace
            },
            "spec": {
                "definitionRef": task.__class__.__name__,    # References UnitTaskDefinition
                "config": task.model_dump(mode='json'),      # Task config as JSON-serializable dict
                "executionDescriptorName": task.execution_descriptor_name  # Optional descriptor name
            }
        }

        try:
            result = await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural="unittaskinstances",
                body=cr_body
            )
            logger.info("Created UnitTaskInstance CR: %s for task: %s", cr_name, task.__class__.__name__)
            return result, cr_name
        except client.ApiException as e:
            if e.status == 409:
                logger.info("UnitTaskInstance %s already exists", cr_name)
                raise ValueError(f"UnitTaskInstance {cr_name} already exists") from e
            raise

    async def get_unit_task_from_instance_cr(
        self,
        cr_name: str,
        namespace: str = K8S_NAMESPACE
    ) -> Any:  # Returns UnitTask object - using Any to avoid circular import
        """Reconstruct a UnitTask object from a UnitTaskInstance CR."""
        await self._load_config()
        assert self._custom_api is not None
        
        try:
            # Download the CR
            cr_object = await self._custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural="unittaskinstances",
                name=cr_name
            )

            spec = cr_object.get("spec", {})
            definition_ref = spec.get("definitionRef")
            config = spec.get("config")

            if not definition_ref:
                raise ValueError(f"UnitTaskInstance CR {cr_name} missing definitionRef")

            if not config:
                raise ValueError(f"UnitTaskInstance CR {cr_name} missing config")

            # Reconstruct the task object using the definition reference and config
            task_cls, _, _ = TASK_REGISTRY.get(definition_ref)
            task_instance = task_cls.model_validate(config)  # config is already a dict, not JSON string

            logger.info("Successfully reconstructed %s from UnitTaskInstance CR: %s",
                       definition_ref, cr_name)
            return task_instance
            
        except client.ApiException as e:
            if e.status == 404:
                raise ValueError(f"UnitTaskInstance CR {cr_name} not found") from e
            raise RuntimeError(f"Failed to get UnitTaskInstance CR {cr_name}: {e}") from e

    async def create_execution_descriptor(
        self,
        name: str,
        task_class_name: str,
        descriptor_class_name: str,
        module_name: str,
        source_code: str,
        is_default: bool = True,
        namespace: str = K8S_NAMESPACE
    ) -> dict[str, Any]:
        """Create an ExecutionDescriptor custom resource."""
        await self._load_config()
        assert self._custom_api is not None

        # Base64 encode the source code
        encoded_source = base64.b64encode(source_code.encode('utf-8')).decode('utf-8')
        
        cr_body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": "ExecutionDescriptor",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "taskClassName": task_class_name,
                "descriptorClassName": descriptor_class_name,
                "moduleName": module_name,
                "sourceCode": encoded_source,
                "isDefault": is_default
            }
        }

        try:
            result = await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural="executiondescriptors",
                body=cr_body
            )
            logger.info("Created ExecutionDescriptor: %s", name)
            return result
        except client.ApiException as e:
            if e.status == 409:
                logger.info("ExecutionDescriptor %s already exists", name)
                raise ValueError(f"ExecutionDescriptor {name} already exists") from e
            raise

    async def get_unit_task_definition(self, name: str, namespace: str = K8S_NAMESPACE) -> dict[str, Any]:
        """Get a UnitTaskDefinition by name."""
        await self._load_config()
        assert self._custom_api is not None
        
        return await self._custom_api.get_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural="unittaskdefinitions",
            name=name
        )

    async def get_unit_task_instance(self, name: str, namespace: str = K8S_NAMESPACE) -> dict[str, Any]:
        """Get a UnitTaskInstance by name."""
        await self._load_config()
        assert self._custom_api is not None
        
        return await self._custom_api.get_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural="unittaskinstances",
            name=name
        )

    async def get_execution_descriptor(self, name: str, namespace: str = K8S_NAMESPACE) -> dict[str, Any]:
        """Get an ExecutionDescriptor by name."""
        await self._load_config()
        assert self._custom_api is not None
        
        return await self._custom_api.get_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural="executiondescriptors",
            name=name
        )

    async def delete_unit_task_definition(self, name: str, namespace: str = K8S_NAMESPACE) -> None:
        """Delete a UnitTaskDefinition by name."""
        await self._load_config()
        assert self._custom_api is not None
        
        await self._custom_api.delete_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural="unittaskdefinitions",
            name=name
        )
        logger.info("Deleted UnitTaskDefinition: %s", name)

    async def delete_unit_task_instance(self, name: str, namespace: str = K8S_NAMESPACE) -> None:
        """Delete a UnitTaskInstance by name."""
        await self._load_config()
        assert self._custom_api is not None
        
        await self._custom_api.delete_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural="unittaskinstances",
            name=name
        )
        logger.info("Deleted UnitTaskInstance: %s", name)

    async def delete_execution_descriptor(self, name: str, namespace: str = K8S_NAMESPACE) -> None:
        """Delete an ExecutionDescriptor by name."""
        await self._load_config()
        assert self._custom_api is not None
        
        await self._custom_api.delete_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural="executiondescriptors",
            name=name
        )
        logger.info("Deleted ExecutionDescriptor: %s", name)

    async def watch_cr_updates(self) -> None:
        """Watch for updates on all Cornserve CRs and process them.
        
        This method watches UnitTaskDefinition, UnitTaskInstance, and ExecutionDescriptor
        resources and handles events by loading tasks into the TASK_REGISTRY as appropriate.
        """
        await self._load_config()
        watchers = [
            self._watch_unittaskdefinitions(),
            self._watch_unittaskinstances(),
            self._watch_executiondescriptors(),
        ]
        await asyncio.gather(*watchers)

    def _handle_cr_event(self, cr_object: dict[str, Any], kind: str, event_type: str = "UPDATE") -> None:
        """Handle CR events by processing the object according to its type."""
        spec = cr_object.get("spec", {})
        metadata = cr_object.get("metadata", {})
        name = metadata.get("name")

        # Handle UnitTaskDefinition CRs by registering them with TASK_REGISTRY
        if kind == "UnitTaskDefinition" and event_type in ("EXISTING", "ADDED", "MODIFIED"):
            try:
                task_class_name = spec.get("taskClassName")
                module_name = spec.get("moduleName")
                source_code = spec.get("sourceCode")
                
                if not task_class_name:
                    logger.error("UnitTaskDefinition %s missing taskClassName", name)
                    return
                
                if not module_name:
                    logger.error("UnitTaskDefinition %s missing moduleName", name)
                    return
                
                if not source_code:
                    logger.error("UnitTaskDefinition %s missing sourceCode", name)
                    return
                
                logger.info("Registering UnitTask %s from CR %s into module %s", task_class_name, name, module_name)
                TASK_REGISTRY.load_from_cr(
                    source_code=source_code,
                    task_class_name=task_class_name,
                    module_name=module_name
                )
                logger.info("Successfully registered UnitTask: %s", task_class_name)
                
            except Exception as e:
                logger.error("Failed to register UnitTask %s from CR %s: %s", 
                           task_class_name if 'task_class_name' in locals() else 'unknown', name, e)
        
        # Handle deletion events by logging (we don't unregister for now)
        elif kind == "UnitTaskDefinition" and event_type == "DELETED":
            task_class_name = spec.get("taskClassName", "unknown")
            logger.info("UnitTaskDefinition %s (class: %s) was deleted. Tasks remain registered.", 
                       name, task_class_name)
        
        # Handle ExecutionDescriptor events
        elif kind == "ExecutionDescriptor" and event_type in ("EXISTING", "ADDED", "MODIFIED"):
            try:
                descriptor_class_name = spec.get("descriptorClassName")
                module_name = spec.get("moduleName")
                source_code = spec.get("sourceCode")
                task_class_name = spec.get("taskClassName")
                
                if not descriptor_class_name:
                    logger.error("ExecutionDescriptor %s missing descriptorClassName", name)
                    return
                
                if not module_name:
                    logger.error("ExecutionDescriptor %s missing moduleName", name)
                    return
                
                if not source_code:
                    logger.error("ExecutionDescriptor %s missing sourceCode", name)
                    return
                
                if not task_class_name:
                    logger.error("ExecutionDescriptor %s missing taskClassName", name)
                    return
                
                logger.info("Registering ExecutionDescriptor %s from CR %s into module %s for task %s", 
                           descriptor_class_name, name, module_name, task_class_name)
                
                from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY
                DESCRIPTOR_REGISTRY.load_from_cr(
                    source_code=source_code,
                    descriptor_class_name=descriptor_class_name,
                    module_name=module_name,
                    task_class_name=task_class_name
                )
                logger.info("Successfully registered ExecutionDescriptor: %s", descriptor_class_name)
                
            except Exception as e:
                logger.error("Failed to register ExecutionDescriptor %s from CR %s: %s", 
                           descriptor_class_name if 'descriptor_class_name' in locals() else 'unknown', name, e)
        
        # Handle ExecutionDescriptor deletion events by logging (we don't unregister for now)
        elif kind == "ExecutionDescriptor" and event_type == "DELETED":
            descriptor_class_name = spec.get("descriptorClassName", "unknown")
            logger.info("ExecutionDescriptor %s (class: %s) was deleted. Descriptors remain registered.", 
                       name, descriptor_class_name)

    async def _watch_resource(self, plural: str, kind: str) -> None:
        """Watch function with the list-then-watch pattern.
        
        Uses list-then-watch pattern:
        1. First lists all existing resources
        2. Processes them as "EXISTING" events  
        3. Then starts watching from the resource version returned by list
        4. This ensures no events are missed between list and watch operations
        """
        assert self._custom_api is not None
        
        while True:  # Retry loop for resilience
            try:
                logger.info("Starting robust watch for %s resources", kind)
                
                # Step 1: List all existing resources to get current state
                initial_list = await self._custom_api.list_namespaced_custom_object(
                    group=CRD_GROUP,
                    version=CRD_VERSION,
                    namespace=K8S_NAMESPACE,
                    plural=plural,
                )
                
                # Step 2: Process existing resources as "EXISTING" events
                existing_items = initial_list.get("items", [])
                logger.info("Found %d existing %s resources", len(existing_items), kind)
                
                for item in existing_items:
                    self._handle_cr_event(item, kind, "EXISTING")
                
                # Step 3: Get the resource version to start watching from
                resource_version = initial_list.get("metadata", {}).get("resourceVersion")
                logger.info("Starting watch for %s from resource version: %s", kind, resource_version)
                
                # Step 4: Start watching from this resource version
                w = Watch()
                async with w.stream(
                    self._custom_api.list_namespaced_custom_object,
                    group=CRD_GROUP,
                    version=CRD_VERSION,
                    namespace=K8S_NAMESPACE,
                    plural=plural,
                    watch=True,
                    resource_version=resource_version,
                    timeout_seconds=300,  # 5 minute timeout to periodically refresh
                ) as stream:
                    async for event in stream:
                        event_type = event.get("type", "UNKNOWN")
                        logger.info("Received watch event: %s for %s", event_type, kind)
                        self._handle_cr_event(event["object"], kind, event_type)
                        
            except asyncio.CancelledError:
                logger.info("Watch for %s was cancelled", kind)
                raise
            except Exception as e:
                logger.error("Error in robust watch for %s: %s", kind, e)
                logger.info("Retrying watch for %s in 5 seconds...", kind)
                await asyncio.sleep(5)
                # Continue the while loop to retry

    async def _watch_unittaskdefinitions(self) -> None:
        """Watch for UnitTaskDefinition updates."""
        await self._watch_resource("unittaskdefinitions", "UnitTaskDefinition")

    async def _watch_unittaskinstances(self) -> None:
        """Watch for UnitTaskInstance updates."""
        await self._watch_resource("unittaskinstances", "UnitTaskInstance")

    async def _watch_executiondescriptors(self) -> None:
        """Watch for ExecutionDescriptor updates."""
        await self._watch_resource("executiondescriptors", "ExecutionDescriptor")

    async def close(self) -> None:
        """Close the API client connection."""
        if self._api_client:
            await self._api_client.close()
            self._api_client = None
            self._custom_api = None
