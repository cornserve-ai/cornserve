"""Task manager that manages registered and deployed tasks."""

from __future__ import annotations

import asyncio
import os
import hashlib
import json
import enum
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import aiohttp
import grpc
from grpc.aio import AioRpcError
from kubernetes_asyncio import client
from kubernetes_asyncio.watch import Watch
from opentelemetry import trace

from cornserve.constants import (
    CR_KEY_MAX_UNIT_TASK_INSTANCE_RV,
    CR_NAME_LATEST_UNIT_TASK_INSTANCE_RV,
    CRD_GROUP,
    CRD_KIND_LATEST_UNIT_TASK_INSTANCE_RV,
    CRD_PLURAL_LATEST_UNIT_TASK_INSTANCE_RVS,
    CRD_PLURAL_UNIT_TASK_INSTANCES,
    CRD_VERSION,
    K8S_NAMESPACE,
    K8S_TASK_DISPATCHER_HTTP_URL,
    REFCOUNT_LEASE_ACQUIRE_TIMEOUT_SECONDS,
    REFCOUNT_LEASE_DURATION_SECONDS,
    REFCOUNT_LEASE_RETRY_INTERVAL_SECONDS,
    SYNC_WATCHERS_POLL_INTERVAL,
)
from cornserve.logging import get_logger
from cornserve.services.pb import common_pb2
from cornserve.services.pb.resource_manager_pb2 import (
    DeployUnitTaskRequest,
    ScaleUnitTaskRequest,
    TeardownUnitTaskRequest,
)
from cornserve.services.pb.resource_manager_pb2_grpc import ResourceManagerStub
from cornserve.services.task_registry import TaskRegistry
from cornserve.task.base import (
    TASK_TIMEOUT,
    MacroUnitTask,
    TaskGraphDispatch,
    TaskInvocation,
    UnitTask,
)
from cornserve.utils import format_grpc_error

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class TaskState(enum.StrEnum):
    """Possible states of a task."""

    # Task is currently being deployed
    DEPLOYING = "not ready"

    # Task is ready to be invoked
    READY = "ready"

    # Task is currently being torn down
    TEARING_DOWN = "tearing down"


class TaskManager:
    """Manages registered and deployed tasks."""

    def __init__(self, resource_manager_grpc_url: str, task_registry: TaskRegistry) -> None:
        """Initialize the task manager.

        Args:
            resource_manager_grpc_url: The gRPC URL of the resource manager.
            task_registry: Registry service for handling task instances by name.
        """
        # A big lock to protect all task states
        self.task_lock = asyncio.Lock()

        # HTTP client
        self.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=TASK_TIMEOUT),
            connector=aiohttp.TCPConnector(limit=0),
        )

        # Task-related state. Key is the task ID.
        self.tasks: dict[str, UnitTask] = {}
        self.task_states: dict[str, TaskState] = {}  # Can be read without holding lock.
        self.unit_task_instance_names: dict[str, str] = {}  # Map task_id -> unit task instance name
        self.task_uuids: dict[str, str] = {}  # Map task_id -> UUID, used to generate CR names
        self.task_invocation_tasks: dict[str, list[asyncio.Task]] = defaultdict(list)
        self.task_usage_counter: dict[str, int] = defaultdict(int)

        # CR Manager for creating/managing unit task instance CRs
        self.task_registry = task_registry

        # gRPC client for resource manager
        self.resource_manager_channel = grpc.aio.insecure_channel(resource_manager_grpc_url)
        self.resource_manager = ResourceManagerStub(self.resource_manager_channel)

        pod_name = os.environ.get("HOSTNAME", "gateway")
        self._lease_holder_identity = f"{pod_name}-{uuid.uuid4().hex[:8]}"
        self._lease_duration_seconds = REFCOUNT_LEASE_DURATION_SECONDS
        self._lease_acquire_timeout_seconds = REFCOUNT_LEASE_ACQUIRE_TIMEOUT_SECONDS
        self._lease_retry_interval_seconds = REFCOUNT_LEASE_RETRY_INTERVAL_SECONDS
        self._unit_task_instance_rv: int = 0

    @staticmethod
    def _compute_canonical_task_key(task: UnitTask) -> str:
        """Compute a deterministic hash for a task.

        Excludes the ``id`` field so that equivalent tasks (same config,
        different instance IDs) always produce the same key — matching
        the semantics of ``UnitTask.is_equivalent_to``.
        """
        config_json = json.dumps(task.model_dump(mode="json", exclude={"id"}), sort_keys=True)
        data = f"{task.__class__.__name__}{config_json}{task.execution_descriptor_name}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def _get_custom_api(self) -> client.CustomObjectsApi:
        """Get a properly-typed CustomObjectsApi from the task registry."""
        await self.task_registry._load_config()
        api = self.task_registry._custom_api
        assert api is not None
        return api

    async def _get_coordination_api(self) -> Any:
        await self.task_registry._load_config()
        api_client = self.task_registry._api_client
        assert api_client is not None
        return client.CoordinationV1Api(api_client)

    @staticmethod
    def _lease_name_for_lock_key(lock_key: str) -> str:
        digest = hashlib.sha256(lock_key.encode()).hexdigest()[:20]
        return f"ut-refcnt-{digest}"

    async def _acquire_refcount_lease(self, lock_key: str) -> str:
        coordination_api = await self._get_coordination_api()
        lease_name = self._lease_name_for_lock_key(lock_key)
        deadline = asyncio.get_running_loop().time() + self._lease_acquire_timeout_seconds

        while True:
            now = datetime.now(timezone.utc)
            create_body = client.V1Lease(
                metadata=client.V1ObjectMeta(name=lease_name, namespace=K8S_NAMESPACE),
                spec=client.V1LeaseSpec(
                    holder_identity=self._lease_holder_identity,
                    lease_duration_seconds=self._lease_duration_seconds,
                    acquire_time=now,
                    renew_time=now,
                    lease_transitions=0,
                ),
            )

            try:
                await coordination_api.create_namespaced_lease(namespace=K8S_NAMESPACE, body=create_body)  # pyright: ignore[reportGeneralTypeIssues]
                return lease_name
            except client.ApiException as e:
                if e.status != 409:
                    raise

            try:
                current: Any = await coordination_api.read_namespaced_lease(  # pyright: ignore[reportGeneralTypeIssues]
                    name=lease_name,
                    namespace=K8S_NAMESPACE,
                )
            except client.ApiException as e:
                if e.status == 404:
                    await asyncio.sleep(self._lease_retry_interval_seconds)
                    continue
                raise

            spec = current.spec or client.V1LeaseSpec()
            renew_time = spec.renew_time
            if renew_time is not None and renew_time.tzinfo is None:
                renew_time = renew_time.replace(tzinfo=timezone.utc)
            lease_duration = spec.lease_duration_seconds or self._lease_duration_seconds
            is_expired = renew_time is None or (now - renew_time).total_seconds() > lease_duration
            held_by_self = spec.holder_identity == self._lease_holder_identity

            if held_by_self or is_expired:
                current_meta = current.metadata or client.V1ObjectMeta(name=lease_name, namespace=K8S_NAMESPACE)
                replacement = client.V1Lease(
                    metadata=client.V1ObjectMeta(
                        name=lease_name,
                        namespace=K8S_NAMESPACE,
                        resource_version=current_meta.resource_version,
                    ),
                    spec=client.V1LeaseSpec(
                        holder_identity=self._lease_holder_identity,
                        lease_duration_seconds=self._lease_duration_seconds,
                        acquire_time=spec.acquire_time if held_by_self and spec.acquire_time else now,
                        renew_time=now,
                        lease_transitions=(spec.lease_transitions or 0) + (0 if held_by_self else 1),
                    ),
                )
                try:
                    await coordination_api.replace_namespaced_lease(  # pyright: ignore[reportGeneralTypeIssues]
                        name=lease_name,
                        namespace=K8S_NAMESPACE,
                        body=replacement,
                    )
                    return lease_name
                except client.ApiException as e:
                    if e.status != 409:
                        raise

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Timed out acquiring refcount lease {lease_name} for key={lock_key}")
            await asyncio.sleep(self._lease_retry_interval_seconds)

    async def _release_refcount_lease(self, lease_name: str) -> None:
        coordination_api = await self._get_coordination_api()
        try:
            current: Any = await coordination_api.read_namespaced_lease(  # pyright: ignore[reportGeneralTypeIssues]
                name=lease_name,
                namespace=K8S_NAMESPACE,
            )
        except client.ApiException as e:
            if e.status == 404:
                return
            raise

        spec = current.spec
        if spec is None or spec.holder_identity != self._lease_holder_identity:
            return

        now = datetime.now(timezone.utc)
        current_meta = current.metadata or client.V1ObjectMeta(name=lease_name, namespace=K8S_NAMESPACE)
        release_body = client.V1Lease(
            metadata=client.V1ObjectMeta(
                name=lease_name,
                namespace=K8S_NAMESPACE,
                resource_version=current_meta.resource_version,
            ),
            spec=client.V1LeaseSpec(
                holder_identity="",
                lease_duration_seconds=1,
                acquire_time=spec.acquire_time or now,
                renew_time=now,
                lease_transitions=spec.lease_transitions or 0,
            ),
        )

        try:
            await coordination_api.replace_namespaced_lease(  # pyright: ignore[reportGeneralTypeIssues]
                name=lease_name,
                namespace=K8S_NAMESPACE,
                body=release_body,
            )
        except client.ApiException as e:
            if e.status not in (404, 409):
                raise

    @asynccontextmanager
    async def _refcount_lock(self, lock_key: str) -> AsyncIterator[None]:
        lease_name = await self._acquire_refcount_lease(lock_key)
        try:
            yield
        finally:
            await self._release_refcount_lease(lease_name)

    # ------------------------------------------------------------------
    # LatestUnitTaskInstanceRV — cross-replica watch sync barrier
    # ------------------------------------------------------------------

    async def ensure_latest_unit_task_instance_rv_cr_exists(self) -> None:
        """Ensure the LatestUnitTaskInstanceRV singleton CR exists (idempotent)."""
        custom_api = await self._get_custom_api()
        try:
            await custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_LATEST_UNIT_TASK_INSTANCE_RVS,
                name=CR_NAME_LATEST_UNIT_TASK_INSTANCE_RV,
            )
            return
        except client.ApiException as e:
            if getattr(e, "status", None) != 404:
                raise

        body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": CRD_KIND_LATEST_UNIT_TASK_INSTANCE_RV,
            "metadata": {"name": CR_NAME_LATEST_UNIT_TASK_INSTANCE_RV, "namespace": K8S_NAMESPACE},
            "spec": {CR_KEY_MAX_UNIT_TASK_INSTANCE_RV: 0},
        }
        try:
            await custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_LATEST_UNIT_TASK_INSTANCE_RVS,
                body=body,
            )
            logger.info("Created LatestUnitTaskInstanceRV singleton CR.")
        except client.ApiException as e:
            if getattr(e, "status", None) == 409:
                return  # another replica beat us
            raise

    async def _publish_unit_task_instance_rv(self, rv: int) -> None:
        """Publish a UnitTaskInstance CR resource version to the singleton.

        The RV must come from a specific CR event (create, modify, or delete
        response) — NOT from a list RV, which is a global etcd RV that the
        per-resource watch may never reach.

        Uses read-then-max to avoid overwriting a higher value from another replica.
        """
        custom_api = await self._get_custom_api()
        rv = int(rv)
        try:
            cr: Any = await custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_LATEST_UNIT_TASK_INSTANCE_RVS,
                name=CR_NAME_LATEST_UNIT_TASK_INSTANCE_RV,
            )
            current = int(cr.get("spec", {}).get(CR_KEY_MAX_UNIT_TASK_INSTANCE_RV, 0))
            if rv <= current:
                return  # another replica already published a higher RV
        except Exception:
            pass  # proceed with write

        patch_body = [
            {"op": "replace", "path": f"/spec/{CR_KEY_MAX_UNIT_TASK_INSTANCE_RV}", "value": rv},
        ]
        await custom_api.patch_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=K8S_NAMESPACE,
            plural=CRD_PLURAL_LATEST_UNIT_TASK_INSTANCE_RVS,
            name=CR_NAME_LATEST_UNIT_TASK_INSTANCE_RV,
            body=patch_body,
        )

    async def sync_unit_task_instance_watchers(self) -> None:
        """Wait until the local UnitTaskInstance watcher has caught up to the target RV."""
        custom_api = await self._get_custom_api()
        try:
            cr = await custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_LATEST_UNIT_TASK_INSTANCE_RVS,
                name=CR_NAME_LATEST_UNIT_TASK_INSTANCE_RV,
            )
            target_rv = int(cr.get("spec", {}).get(CR_KEY_MAX_UNIT_TASK_INSTANCE_RV, 0))
        except Exception:
            return  # singleton doesn't exist yet — nothing to sync

        while self._unit_task_instance_rv < target_rv:
            await asyncio.sleep(SYNC_WATCHERS_POLL_INTERVAL)

    # ------------------------------------------------------------------

    async def _get_usage_refcount(self, instance_name: str) -> int:
        """Read the current refcount from the CR."""
        custom_api = await self._get_custom_api()
        try:
            cr: Any = await custom_api.get_namespaced_custom_object(  # pyright: ignore[reportGeneralTypeIssues]
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
                name=instance_name,
            )
            spec = cr.get("spec", {})
            return int(spec.get("usageRefcount", 1))
        except Exception as e:
            logger.warning("Failed to get usage refcount for %s, defaulting to 1: %s", instance_name, e)
            return 1

    async def _patch_usage_refcount(self, instance_name: str, new_value: int) -> None:
        """Update the usageRefcount field on the CR using JSON patch."""
        custom_api = await self._get_custom_api()
        patch_body = [{"op": "replace", "path": "/spec/usageRefcount", "value": new_value}]
        await custom_api.patch_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=K8S_NAMESPACE,
            plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
            name=instance_name,
            body=patch_body,
        )

    async def _get_or_create_task_instance_with_refcount(self, task: UnitTask) -> tuple[str, bool, int]:
        """Check for existing CR or create a new one, and manage its refcount.

        Returns:
            (instance_name, already_existed, cr_resource_version).
            The RV is from a specific CR mutation and is safe to publish
            for watch-sync (unlike a list RV which is a global etcd RV).
        """
        custom_api = await self._get_custom_api()

        task_config = task.model_dump(mode="json", exclude={"id"})
        definition_ref = task.__class__.__name__
        execution_descriptor_name = task.execution_descriptor_name
        lock_key = self._compute_canonical_task_key(task)

        async with self._refcount_lock(lock_key):
            try:
                resp: Any = await custom_api.list_namespaced_custom_object(  # pyright: ignore[reportGeneralTypeIssues]
                    group=CRD_GROUP,
                    version=CRD_VERSION,
                    namespace=K8S_NAMESPACE,
                    plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
                )
                items = resp.get("items", [])
                for item in items:
                    spec = item.get("spec", {})
                    cr_config = spec.get("config", {})
                    # Compare excluding 'id' — same semantics as is_equivalent_to
                    cr_config_no_id = {k: v for k, v in cr_config.items() if k != "id"}
                    if (
                        cr_config_no_id == task_config
                        and spec.get("definitionRef") == definition_ref
                        and spec.get("executionDescriptorName") == execution_descriptor_name
                    ):
                        instance_name = item["metadata"]["name"]
                        current_refcount = int(spec.get("usageRefcount", 1))
                        await self._patch_usage_refcount(instance_name, current_refcount + 1)
                        # Read back to get the post-patch RV (safe for watch-sync)
                        updated: Any = await custom_api.get_namespaced_custom_object(
                            group=CRD_GROUP, version=CRD_VERSION,
                            namespace=K8S_NAMESPACE,
                            plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
                            name=instance_name,
                        )
                        cr_rv = int(updated["metadata"]["resourceVersion"])
                        logger.info(
                            "Found existing CR %s for task %s, bumped refcount %d -> %d",
                            instance_name,
                            definition_ref,
                            current_refcount,
                            current_refcount + 1,
                        )
                        return instance_name, True, cr_rv
            except Exception as e:
                logger.warning("Error listing task instances: %s", e)

            task_uuid = self._compute_canonical_task_key(task)
            instance_name = await self.task_registry.create_task_instance_from_task(task, task_uuid)
            await self._patch_usage_refcount(instance_name, 1)
            # Read back the CR to get the post-patch RV
            created: Any = await custom_api.get_namespaced_custom_object(
                group=CRD_GROUP, version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
                name=instance_name,
            )
            cr_rv = int(created["metadata"]["resourceVersion"])
            logger.info("Created new CR %s for task %s with refcount=1", instance_name, definition_ref)
            return instance_name, False, cr_rv

    async def _decrement_usage_refcount(self, instance_name: str) -> int:
        """Decrement the usageRefcount on the CR and return the new value.

        Returns 0 if the CR no longer exists (already deleted by another
        gateway replica or ``clear_all``).
        """
        current = await self._get_usage_refcount(instance_name)
        new_val = max(0, current - 1)
        try:
            await self._patch_usage_refcount(instance_name, new_val)
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                logger.warning("CR %s already deleted, treating refcount as 0", instance_name)
                return 0
            raise
        return new_val

    async def _release_task_reference(self, task: UnitTask, instance_name: str) -> bool:
        lock_key = self._compute_canonical_task_key(task)
        async with self._refcount_lock(lock_key):
            new_cr_refcount = await self._decrement_usage_refcount(instance_name)
            logger.info("Decremented CR refcount for %s: new refcount=%d", instance_name, new_cr_refcount)
            if new_cr_refcount > 0:
                return False

            # Teardown the task manager (free GPUs, kill pods).
            # "not found" / "not running" means another replica already
            # tore it down — treat as success.
            try:
                response = await self.resource_manager.TeardownUnitTask(
                    TeardownUnitTaskRequest(task_instance_name=instance_name)
                )
                if response.status != common_pb2.Status.STATUS_OK:
                    logger.warning("TeardownUnitTask for %s returned non-OK: %s", instance_name, response)
            except Exception as e:
                err_str = str(e).lower()
                if "not found" in err_str or "not running" in err_str:
                    logger.info("TeardownUnitTask for %s: already gone, treating as success", instance_name)
                else:
                    logger.warning("TeardownUnitTask for %s failed: %s", instance_name, e)

            # Delete the CR.  This runs inside _refcount_lock (distributed
            # lease) so it's serialized with _get_or_create on other replicas.
            try:
                delete_rv = await self.task_registry.delete_task_instance(instance_name)
                logger.info("Deleted CR %s after refcount reached 0 (rv=%d)", instance_name, delete_rv)
                # Publish the deletion RV so other replicas' list_tasks syncs
                # past it (otherwise they may still see the stale entry).
                if delete_rv > 0:
                    await self._publish_unit_task_instance_rv(delete_rv)
            except Exception as e:
                logger.warning("Failed to delete CR %s (may already be gone): %s", instance_name, e)
            return True

    async def _ensure_tasks_from_crs(self) -> None:
        """Query existing UnitTaskInstance CRs from k8s and populate local state.

        For each CR that exists in K8s, ensures a READY entry is present in
        local state.  Existing entries whose state is already READY are left
        untouched (preserving the task_id assigned by ``declare_used``).
        """
        custom_api = await self._get_custom_api()
        try:
            resp: Any = await custom_api.list_namespaced_custom_object(  # pyright: ignore[reportGeneralTypeIssues]
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
            )
            items = resp.get("items", [])
            for item in items:
                instance_name = item["metadata"]["name"]

                async with self.task_lock:
                    # Check if we already have a READY entry for this CR.
                    existing_tid = None
                    for tid, name in self.unit_task_instance_names.items():
                        if name == instance_name:
                            existing_tid = tid
                            break
                    if existing_tid is not None and self.task_states.get(existing_tid) == TaskState.READY:
                        continue  # Already present and healthy

                try:
                    reconstructed_task = await self.task_registry.get_task_instance(instance_name)
                    task_id = instance_name
                    uuid_part = instance_name.split("-", 1)[1] if "-" in instance_name else instance_name

                    spec = item.get("spec", {})
                    refcount = int(spec.get("usageRefcount", 1))

                    async with self.task_lock:
                        if task_id not in self.tasks:
                            self.tasks[task_id] = reconstructed_task
                            self.task_states[task_id] = TaskState.READY
                            self.unit_task_instance_names[task_id] = instance_name
                            self.task_uuids[task_id] = uuid_part
                            self.task_usage_counter[task_id] = refcount
                            logger.info("Reconstructed and registered task %s from CR", task_id)
                except Exception as e:
                    logger.warning("Failed to reconstruct task %s: %s", instance_name, e)
                    continue
        except Exception as e:
            logger.error("Error listing task instances from k8s: %s", e)

    async def watch_task_instances(self) -> None:
        """Background task: list + watch UnitTaskInstance CRs and maintain local state."""
        custom_api = await self._get_custom_api()

        while True:
            try:
                if self._unit_task_instance_rv == 0:
                    initial_list = await custom_api.list_namespaced_custom_object(
                        group=CRD_GROUP,
                        version=CRD_VERSION,
                        namespace=K8S_NAMESPACE,
                        plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
                    )
                    for item in initial_list.get("items", []):
                        await self._handle_task_instance_event(item, "EXISTING")
                    self._unit_task_instance_rv = int(initial_list["metadata"]["resourceVersion"])

                async with Watch().stream(
                    custom_api.list_namespaced_custom_object,
                    group=CRD_GROUP,
                    version=CRD_VERSION,
                    namespace=K8S_NAMESPACE,
                    plural=CRD_PLURAL_UNIT_TASK_INSTANCES,
                    watch=True,
                    resource_version=str(self._unit_task_instance_rv),
                    timeout_seconds=300,
                ) as stream:
                    async for event in stream:
                        obj = event["object"]
                        await self._handle_task_instance_event(obj, event.get("type", "UNKNOWN"))
                        self._unit_task_instance_rv = int(obj["metadata"]["resourceVersion"])

            except asyncio.CancelledError:
                raise
            except client.ApiException as e:
                if getattr(e, "status", None) == 410:
                    logger.warning("UnitTaskInstance watch expired (410 Gone). Relisting.")
                    self._unit_task_instance_rv = 0
                    continue
                logger.error("Error watching UnitTaskInstance (API): %s", e)
                await asyncio.sleep(5)
            except Exception as e:
                logger.error("Error watching UnitTaskInstance: %s", e)
                await asyncio.sleep(5)

    async def _handle_task_instance_event(self, obj: dict[str, Any], event_type: str) -> None:
        """Process a single UnitTaskInstance watch event."""
        instance_name = obj["metadata"]["name"]

        if event_type in ("EXISTING", "ADDED", "MODIFIED"):
            async with self.task_lock:
                if instance_name in self.unit_task_instance_names.values():
                    return  # Locally managed, skip

            try:
                reconstructed_task = await self.task_registry.get_task_instance(instance_name)
            except Exception as e:
                logger.warning("Watch: failed to reconstruct task from CR %s: %s", instance_name, e)
                return

            task_id = instance_name
            uuid_part = instance_name.split("-", 1)[1] if "-" in instance_name else instance_name

            async with self.task_lock:
                # Re-check: another path (e.g. declare_used Phase 3) may have
                # committed this instance_name while we were reconstructing.
                if task_id not in self.tasks and instance_name not in self.unit_task_instance_names.values():
                    self.tasks[task_id] = reconstructed_task
                    self.task_states[task_id] = TaskState.READY
                    self.unit_task_instance_names[task_id] = instance_name
                    self.task_uuids[task_id] = uuid_part
                    logger.info("Watch: synced task %s from CR %s", task_id, instance_name)

        elif event_type == "DELETED":
            async with self.task_lock:
                found_task_id = None
                for tid, name in self.unit_task_instance_names.items():
                    if name == instance_name:
                        found_task_id = tid
                        break

                if found_task_id is not None:
                    self.tasks.pop(found_task_id, None)
                    self.task_states.pop(found_task_id, None)
                    self.unit_task_instance_names.pop(found_task_id, None)
                    self.task_uuids.pop(found_task_id, None)
                    self.task_usage_counter.pop(found_task_id, None)
                    for it in self.task_invocation_tasks.pop(found_task_id, []):
                        it.cancel()
                    logger.info("Watch: removed task %s (CR %s deleted)", found_task_id, instance_name)

    async def declare_used(self, tasks: list[UnitTask]) -> None:
        """Deploy the given tasks.

        If a task is already deployed locally, it will be skipped.
        For tasks not yet deployed locally, we check if an equivalent
        UnitTaskInstance CR already exists (deployed by another gateway
        replica). If so, we increment the CR's usageRefcount and skip
        gRPC deployment. Otherwise, we create a new CR and deploy via gRPC.

        An error raised during deployment will roll back the deployment of all tasks deployed.

        The lock is held only for state bookkeeping; slow I/O (task-instance
        creation and gRPC deploy) runs outside the lock so that multiple
        ``declare_used`` calls can proceed concurrently.
        """
        logger.info("Declaring tasks as used: %r", tasks)

        # ------------------------------------------------------------------
        # Phase 1 (locked): register intent — mark new tasks as DEPLOYING,
        # increment usage counters, record what we added for rollback.
        # ------------------------------------------------------------------
        task_ids: list[str] = []
        to_deploy: list[str] = []
        newly_added: list[str] = []       # task_ids we inserted (for rollback)
        incremented: list[str] = []        # task_ids whose counter we bumped
        to_bump_refcount: list[str] = []  # task_ids needing CR refcount bump (watch-synced, first local use)

        async with self.task_lock:
            for task in tasks:
                # Check if the task is already deployed
                for task_id, existing_task in self.tasks.items():
                    if existing_task.is_equivalent_to(task):
                        logger.info("Skipping already deployed task: %r", task)
                        task_ids.append(task_id)
                        # If this replica hasn't claimed this task yet (watch-synced),
                        # we need to bump the CR refcount in Phase 2.
                        if self.task_usage_counter[task_id] == 0:
                            to_bump_refcount.append(task_id)
                        break
                else:
                    # If the task is not already deployed, deploy it
                    logger.info("Should deploy task: %r", task)

                    # Generate a unique ID for the task
                    while True:
                        task_uuid = uuid.uuid4().hex
                        task_id = f"{task.__class__.__name__.lower()}-{task_uuid}"
                        if task_id not in self.tasks:
                            break

                    self.tasks[task_id] = task
                    self.task_states[task_id] = TaskState.DEPLOYING
                    self.task_uuids[task_id] = task_uuid
                    task_ids.append(task_id)
                    to_deploy.append(task_id)
                    newly_added.append(task_id)

                # Whether or not it was already deployed, increment the usage counter
                self.task_usage_counter[task_id] += 1
                incremented.append(task_id)

        # ------------------------------------------------------------------
        # Phase 2 (unlocked): check for existing CRs, create if needed,
        # and gRPC deploy only truly new tasks.
        # ------------------------------------------------------------------
        unit_task_instance_names: dict[str, str] = {}
        tasks_needing_grpc_deploy: list[str] = []  # subset of to_deploy
        tasks_already_deployed_in_cluster: list[str] = []  # CR exists, skip gRPC
        max_cr_rv = 0
        try:
            for task_id in to_deploy:
                task = self.tasks[task_id]

                # Check for existing equivalent CR or create a new one,
                # and atomically manage the usageRefcount on the CR.
                instance_name, already_existed, cr_rv = await self._get_or_create_task_instance_with_refcount(task)
                unit_task_instance_names[task_id] = instance_name
                max_cr_rv = max(max_cr_rv, cr_rv)

                if already_existed:
                    # CR exists — another gateway already deployed this task.
                    # No need for gRPC deploy; just adopt it locally.
                    logger.info(
                        "Task %s already deployed in cluster (CR %s), skipping gRPC deploy",
                        task_id, instance_name,
                    )
                    tasks_already_deployed_in_cluster.append(task_id)
                else:
                    tasks_needing_grpc_deploy.append(task_id)

            # gRPC deploy only truly new tasks
            coros = []
            for task_id in tasks_needing_grpc_deploy:
                coros.append(
                    self.resource_manager.DeployUnitTask(
                        DeployUnitTaskRequest(task_instance_name=unit_task_instance_names[task_id])
                    )
                )

            if coros:
                responses = await asyncio.gather(*coros, return_exceptions=True)

                # Check for deployment errors
                errors: list[BaseException] = []
                for resp, deployed_task in zip(responses, tasks_needing_grpc_deploy, strict=True):
                    if isinstance(resp, AioRpcError):
                        logger.exception("gRPC error while deploying task %s: \n%s", deployed_task, format_grpc_error(resp))
                        errors.append(resp)
                    elif isinstance(resp, BaseException):
                        logger.exception("Error while deploying task: %s", resp)
                        errors.append(resp)

                if errors:
                    raise RuntimeError("Error while deploying tasks")

            # Bump CR refcount for watch-synced tasks being used locally for the first time
            for task_id in to_bump_refcount:
                task = self.tasks[task_id]
                instance_name = self.unit_task_instance_names[task_id]
                lock_key = self._compute_canonical_task_key(task)
                async with self._refcount_lock(lock_key):
                    current = await self._get_usage_refcount(instance_name)
                    await self._patch_usage_refcount(instance_name, current + 1)
                    logger.info(
                        "Bumped CR refcount for watch-synced task %s (%s): %d -> %d",
                        task_id, instance_name, current, current + 1,
                    )

        except BaseException as e:
            # Catch BaseException (not just Exception) so that
            # asyncio.CancelledError / GeneratorExit from a dropped SSE
            # connection during /app/register also triggers cleanup.
            # Without this, a cancelled declare_used leaves ghost entries
            # stuck in DEPLOYING with no unit_task_instance_names mapping,
            # which poison invoke_tasks on this replica until pod restart.

            # Teardown any tasks that were sent to the RM (only newly deployed ones)
            cleanup_coros = [
                self.resource_manager.TeardownUnitTask(
                    TeardownUnitTaskRequest(task_instance_name=unit_task_instance_names[tid])
                )
                for tid in tasks_needing_grpc_deploy
                if tid in unit_task_instance_names
            ]
            if cleanup_coros:
                await asyncio.gather(*cleanup_coros, return_exceptions=True)

            # Rollback CR refcount bumps for watch-synced tasks
            for tid in to_bump_refcount:
                try:
                    instance_name = self.unit_task_instance_names.get(tid)
                    if instance_name:
                        task = self.tasks[tid]
                        lock_key = self._compute_canonical_task_key(task)
                        async with self._refcount_lock(lock_key):
                            current = await self._get_usage_refcount(instance_name)
                            await self._patch_usage_refcount(instance_name, max(0, current - 1))
                except Exception:
                    logger.warning("Failed to rollback CR refcount for %s", tid)

            # Targeted rollback: undo only this call's state changes
            async with self.task_lock:
                for tid in newly_added:
                    self.tasks.pop(tid, None)
                    self.task_states.pop(tid, None)
                    self.task_uuids.pop(tid, None)
                    self.unit_task_instance_names.pop(tid, None)
                for tid in incremented:
                    self.task_usage_counter[tid] -= 1
                    if self.task_usage_counter[tid] <= 0:
                        del self.task_usage_counter[tid]
            logger.info("Rolled back deployment: %r", e)
            raise

        # ------------------------------------------------------------------
        # Phase 3 (locked): commit — store instance names, mark READY.
        # ------------------------------------------------------------------
        async with self.task_lock:
            for task_id in to_deploy:
                self.unit_task_instance_names[task_id] = unit_task_instance_names[task_id]

            for task_id in task_ids:
                if task_id not in self.tasks:
                    raise ValueError(f"Task with ID {task_id} does not exist")
                self.task_states[task_id] = TaskState.READY

        # Publish the max CR resource version so other replicas can sync.
        # Publish the CR RV so other replicas' list_tasks can sync.
        if max_cr_rv > 0:
            await self._publish_unit_task_instance_rv(max_cr_rv)
            await self.sync_unit_task_instance_watchers()

    async def declare_not_used(self, tasks: list[UnitTask]) -> None:
        """Declare that the given tasks are not used anymore.

        Every call decrements both the local usage counter and the CR's
        usageRefcount (enabling cross-replica teardown). Local state cleanup
        only happens when the local counter reaches 0. When the CR refcount
        reaches 0, _release_task_reference issues a gRPC teardown and deletes
        the CR.

        If the specific task is not deployed locally, it will be skipped.
        An error raised during tear down will *not* roll back the tear down of other tasks.
        """
        to_release: list[tuple[str, UnitTask, str]] = []
        async with self.task_lock:
            for task in tasks:
                for task_id, existing_task in self.tasks.items():
                    if existing_task.is_equivalent_to(task):
                        usage_counter = self.task_usage_counter.get(task_id, 0)
                        if usage_counter > 0:
                            usage_counter -= 1
                            self.task_usage_counter[task_id] = usage_counter

                        # Always release CR refcount (not just when local counter hits 0)
                        instance_name = self.unit_task_instance_names.get(task_id)
                        if instance_name is None:
                            # Ghost task: declare_used Phase 3 never ran (e.g.
                            # cancelled SSE connection).  No CR exists, so just
                            # purge from local state — there is nothing to tear
                            # down on the RM side.
                            logger.warning(
                                "Purging ghost task %s (no CR name, likely from a "
                                "cancelled deploy)",
                                task_id,
                            )
                            self.tasks.pop(task_id, None)
                            self.task_states.pop(task_id, None)
                            self.task_uuids.pop(task_id, None)
                            self.task_usage_counter.pop(task_id, None)
                            break

                        logger.info("Releasing CR refcount for task %s (local_counter=%d)", task_id, usage_counter)
                        to_release.append((task_id, existing_task, instance_name))

                        # Only mark as TEARING_DOWN and cancel invocations if local counter is 0
                        if usage_counter == 0:
                            self.task_states[task_id] = TaskState.TEARING_DOWN
                            for invocation_task in self.task_invocation_tasks.pop(task_id, []):
                                invocation_task.cancel()
                        break
                else:
                    logger.warning("Cannot find task, skipping teardown: %r", task)

        fully_done: list[str] = []     # local counter=0, remove from local state
        refcount_only: list[str] = []  # local counter>0, keep local state
        errors: list[BaseException] = []
        for task_id, task, instance_name in to_release:
            try:
                did_teardown = await self._release_task_reference(task, instance_name)
                if self.task_usage_counter.get(task_id, 0) == 0:
                    fully_done.append(task_id)
                else:
                    refcount_only.append(task_id)
                    # Restore state since this replica still uses it
                    async with self.task_lock:
                        if task_id in self.task_states:
                            self.task_states[task_id] = TaskState.READY
            except BaseException as e:
                logger.exception("Error while releasing task %s (%s): %s", task_id, instance_name, e)
                errors.append(e)
                # If the release failed due to 404 (CR already deleted by another gateway),
                # treat it as success for local cleanup
                if self.task_usage_counter.get(task_id, 0) == 0:
                    fully_done.append(task_id)

        if fully_done:
            async with self.task_lock:
                for task_id in fully_done:
                    self.tasks.pop(task_id, None)
                    self.task_states.pop(task_id, None)
                    self.unit_task_instance_names.pop(task_id, None)
                    self.task_uuids.pop(task_id, None)
                    self.task_usage_counter.pop(task_id, None)

        for task_id in refcount_only:
            logger.info("Decremented CR refcount for task %s (still in local use)", task_id)
        for task_id in fully_done:
            logger.info("Teardown complete for task %s", task_id)

        if errors:
            logger.error("Errors occurred while tearing down tasks")
            raise RuntimeError(f"Error while tearing down tasks: {errors}")

    async def scale_unit_task(self, task_id: str, num_gpus: int) -> None:
        """Scale the given unit task of task_id to add or remove specified number of GPUs."""
        try:
            if task_id not in self.tasks:
                # One-time CR sync as fallback for watch propagation delay
                await self._ensure_tasks_from_crs()
            if task_id not in self.tasks:
                raise KeyError(f"Unit Task with task_id {task_id} is not deployed")
            if self.task_states[task_id] != TaskState.READY:
                raise RuntimeError(f"Unit Task {task_id} is not ready yet. Retry when it's ready.")
            if task_id not in self.unit_task_instance_names:
                raise RuntimeError(f"No CR name found for task {task_id}")

            unit_task_instance_name = self.unit_task_instance_names[task_id]
            response = await self.resource_manager.ScaleUnitTask(
                ScaleUnitTaskRequest(task_instance_name=unit_task_instance_name, num_gpus=num_gpus)
            )
            if response.status != common_pb2.Status.STATUS_OK:
                raise RuntimeError(f"Failed to scale task {task_id} to update {num_gpus} GPUs: {response.message}")
        except Exception as e:
            logger.exception("Error while scaling unit task %s", task_id)
            raise RuntimeError(f"Error while scaling unit task {task_id}: {e}") from e

    async def list_tasks(self) -> list[tuple[UnitTask, str, TaskState]]:
        await self.sync_unit_task_instance_watchers()
        return [(task, task_id, self.task_states[task_id]) for task_id, task in self.tasks.items()]

    async def invoke_tasks(self, dispatch: TaskGraphDispatch) -> list[Any]:
        """Invoke the given tasks.

        Before invocation, this method ensures that all tasks part of the invocation
        are deployed and ready to be invoked. It is ensured that the number of outputs
        returned by the task dispatcher matches the number of invocations.

        Args:
            dispatch: The dispatch object containing the tasks to invoke.

        Returns:
            The outputs of all tasks.
        """
        expanded_invocations = self._expand_macro_invocations(dispatch.invocations)

        # Resolve all tasks from local cache — no CR fallback needed.
        # Workers are guaranteed to be synced by the gateway-master before
        # the client can reach this point.
        running_task_ids: list[str] = []
        async with self.task_lock:
            for invocation in expanded_invocations:
                # Search for a READY equivalent task.  There may be stale
                # entries left behind by a cancelled ``declare_used`` (e.g.
                # client disconnect during ``/app/register``).  Those ghost
                # entries sit in DEPLOYING with no backing task-manager and
                # no ``unit_task_instance_names`` mapping.  We must skip
                # them and keep looking for the real READY entry that the
                # K8s watch handler added.
                found_ready = False
                found_deploying = False
                for task_id, task in self.tasks.items():
                    if task.is_equivalent_to(invocation.task):
                        match self.task_states[task_id]:
                            case TaskState.READY:
                                running_task_ids.append(task_id)
                                found_ready = True
                                break
                            case TaskState.DEPLOYING:
                                found_deploying = True
                            case TaskState.TEARING_DOWN:
                                pass  # skip, keep looking
                if found_ready:
                    continue
                if found_deploying:
                    raise ValueError(f"Task {invocation.task} is being deployed")
                # No equivalent task found at all
                raise KeyError(f"Task {invocation.task} is not deployed")
        assert len(running_task_ids) == len(expanded_invocations)

        # Dispatch to the Task Dispatcher
        invocation_task = asyncio.create_task(dispatch.dispatch(K8S_TASK_DISPATCHER_HTTP_URL + "/task", self.client))
        # Store the invocation task under the task IDs of all running tasks.
        # If any of the unit tasks are unregistered, the whole thing will be cancelled.
        for task_id in running_task_ids:
            self.task_invocation_tasks[task_id].append(invocation_task)
        try:
            output = await invocation_task
        except asyncio.CancelledError:
            logger.info("Invocation task was cancelled: %s", dispatch)
            raise RuntimeError(
                "Invocation task was cancelled. This is likely because one or more "
                "constituent unit tasks were unregistered.",
            ) from None
        finally:
            # Remove the invocation task from all task IDs
            for task_id in running_task_ids:
                self.task_invocation_tasks[task_id].remove(invocation_task)

        if not isinstance(output, list):
            raise RuntimeError(f"Invalid response from task dispatcher: {output}")
        if len(output) != len(dispatch.invocations):
            raise RuntimeError(f"Expected {len(dispatch.invocations)} outputs, got {len(output)}: {output}")
        return output

    def _expand_macro_invocations(self, invocations: list[TaskInvocation]) -> list[TaskInvocation]:
        expanded_invocations: list[TaskInvocation] = []
        for invocation in invocations:
            if not isinstance(invocation.task, MacroUnitTask):
                expanded_invocations.append(invocation)
                continue

            macro_invocations, _ = invocation.task.expand_invocations(invocation.task_input)
            expanded_invocations.extend(macro_invocations)
        return expanded_invocations

    async def shutdown(self) -> None:
        """Shutdown the task manager."""
        logger.info("Shutting down the Gateway task manager")

        # Close the gRPC channel to the resource manager
        await self.resource_manager_channel.close()

        # Close the HTTP client session
        await self.client.close()

        logger.info("Gateway task manager has been shut down")
