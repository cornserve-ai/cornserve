# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportPossiblyUnboundVariable=false, reportArgumentType=false
"""AppRegistry: CRD-backed distributed app registry for gateway scale-out.

Mirrors the pattern established by TaskRegistry in
``cornserve/services/task_registry/registry.py``:
- lazy k8s client init
- background watcher (list+watch with resourceVersion handling)
- sync barrier (sync_watchers) for registration flow
- graceful shutdown

Each gateway replica runs its own AppRegistry watcher so that app state
converges across all replicas.  The ``AppInstance`` CRs are the source of
truth; each gateway materialises an in-process ``ModuleType`` cache from
the ``sourceCode`` stored in the CR.
"""

from __future__ import annotations

import asyncio
from typing import Any

from kubernetes_asyncio import client, config
from kubernetes_asyncio.watch import Watch

from cornserve.constants import (
    CR_KEY_MAX_APP_RV,
    CR_NAME_LATEST_APP_RV,
    CRD_GROUP,
    CRD_KIND_APP_INSTANCE,
    CRD_KIND_LATEST_APP_RV,
    CRD_PLURAL_APP_INSTANCES,
    CRD_PLURAL_LATEST_APP_RVS,
    CRD_VERSION,
    K8S_NAMESPACE,
    SYNC_WATCHERS_POLL_INTERVAL,
)
from cornserve.logging import get_logger

logger = get_logger(__name__)


class AppRegistry:
    """CRD-backed registry of AppInstance custom resources.

    Provides CRUD helpers and a background watcher that keeps a local
    cache (``self._apps``) in sync with the API server.
    """

    def __init__(self) -> None:
        self._api_client: client.ApiClient | None = None
        self._custom_api: client.CustomObjectsApi | None = None

        # Watcher resource version (updated by background watch task)
        self._app_instance_rv: int = 0

        # Local cache populated by watcher: app_id -> CR spec dict
        self._apps: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Kubernetes client
    # ------------------------------------------------------------------

    async def _load_config(self) -> None:
        if self._api_client:
            return
        try:
            config.load_incluster_config()
            logger.info("AppRegistry: Loaded in-cluster Kubernetes config.")
        except config.ConfigException as e:
            logger.error("AppRegistry: Failed to load Kubernetes config: %s", e)
            raise RuntimeError("Could not load Kubernetes configuration") from e

        self._api_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(self._api_client)

    # ------------------------------------------------------------------
    # LatestAppRV singleton CR helpers
    # ------------------------------------------------------------------

    async def ensure_latest_app_rv_cr_exists(
        self, *, namespace: str = K8S_NAMESPACE, name: str = CR_NAME_LATEST_APP_RV
    ) -> None:
        """Ensure the LatestAppRV singleton CR exists (idempotent)."""
        await self._load_config()
        assert self._custom_api is not None

        try:
            await self._custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural=CRD_PLURAL_LATEST_APP_RVS,
                name=name,
            )
            return  # already exists
        except client.ApiException as e:
            if getattr(e, "status", None) != 404:
                raise

        body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": CRD_KIND_LATEST_APP_RV,
            "metadata": {"name": name, "namespace": namespace},
            "spec": {CR_KEY_MAX_APP_RV: 0},
        }

        try:
            await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural=CRD_PLURAL_LATEST_APP_RVS,
                body=body,
            )
            logger.info("Created LatestAppRV singleton CR.")
        except client.ApiException as e:
            if getattr(e, "status", None) == 409:
                # Another replica beat us — that's fine.
                return
            logger.error("Failed to initialize LatestAppRV singleton CR: %s", e)
            raise

    async def update_latest_app_rv(
        self,
        max_app_rv: int,
        *,
        namespace: str = K8S_NAMESPACE,
        name: str = CR_NAME_LATEST_APP_RV,
    ) -> None:
        """Update the singleton LatestAppRV CR with the latest app RV."""
        await self._load_config()
        assert self._custom_api is not None

        max_app_rv = int(max_app_rv)
        patch_body = [
            {"op": "replace", "path": f"/spec/{CR_KEY_MAX_APP_RV}", "value": max_app_rv},
        ]

        await self._custom_api.patch_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural=CRD_PLURAL_LATEST_APP_RVS,
            name=name,
            body=patch_body,
        )

    async def get_latest_app_rv(
        self,
        namespace: str = K8S_NAMESPACE,
        name: str = CR_NAME_LATEST_APP_RV,
    ) -> int:
        """Read the latest app resource version from the LatestAppRV CR."""
        await self._load_config()
        assert self._custom_api is not None

        try:
            cr = await self._custom_api.get_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural=CRD_PLURAL_LATEST_APP_RVS,
                name=name,
            )
            spec = cr.get("spec", {})
            return int(spec.get(CR_KEY_MAX_APP_RV, 0))
        except Exception as e:
            logger.warning("Failed to read LatestAppRV CR: %s", e)
            return 0

    # ------------------------------------------------------------------
    # AppInstance CR CRUD
    # ------------------------------------------------------------------

    async def create_app_instance(
        self,
        app_id: str,
        source_code: str,
        task_keys: list[str],
        is_streaming: bool,
        state: str,
        *,
        namespace: str = K8S_NAMESPACE,
    ) -> dict[str, Any]:
        """Create an AppInstance CR.  Returns the created object dict.

        The CR name is derived deterministically from ``app_id`` to ensure
        collision-safe naming (a 409 from the API server means the app was
        already registered by another gateway).
        """
        await self._load_config()
        assert self._custom_api is not None

        # Deterministic CR name from app_id (k8s names must be DNS-safe).
        cr_name = app_id  # app_id is already "app-<hex>" which is DNS-safe

        body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": CRD_KIND_APP_INSTANCE,
            "metadata": {"name": cr_name, "namespace": namespace},
            "spec": {
                "appId": app_id,
                "sourceCode": source_code,
                "taskKeys": task_keys,
                "isStreaming": is_streaming,
                "state": state,
            },
        }

        try:
            return await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural=CRD_PLURAL_APP_INSTANCES,
                body=body,
            )
        except client.ApiException as e:
            if e.status == 409:
                raise ValueError(f"App '{app_id}' already registered") from e
            raise

    async def update_app_state(
        self,
        app_id: str,
        state: str,
        *,
        namespace: str = K8S_NAMESPACE,
    ) -> dict[str, Any]:
        """Patch the ``state`` field of an existing AppInstance CR."""
        await self._load_config()
        assert self._custom_api is not None

        patch_body = [{"op": "replace", "path": "/spec/state", "value": state}]

        return await self._custom_api.patch_namespaced_custom_object(
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=namespace,
            plural=CRD_PLURAL_APP_INSTANCES,
            name=app_id,
            body=patch_body,
        )

    async def delete_app_instance(
        self,
        app_id: str,
        *,
        namespace: str = K8S_NAMESPACE,
    ) -> int:
        """Delete an AppInstance CR.

        Returns the resource version of the deleted object (for watch-sync
        tracking), or 0 if the CR was already absent.
        """
        await self._load_config()
        assert self._custom_api is not None

        try:
            resp = await self._custom_api.delete_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural=CRD_PLURAL_APP_INSTANCES,
                name=app_id,
            )
            rv = int(resp.get("metadata", {}).get("resourceVersion", 0)) if isinstance(resp, dict) else 0
            logger.info("Deleted AppInstance CR: %s (rv=%d)", app_id, rv)
            return rv
        except client.ApiException as e:
            if getattr(e, "status", None) == 404:
                logger.info("AppInstance CR already absent: %s", app_id)
                return 0
            logger.error("Failed to delete AppInstance %s: %s", app_id, e)
            raise RuntimeError(f"Failed to delete AppInstance {app_id}: {e}") from e

    # ------------------------------------------------------------------
    # Local cache accessors
    # ------------------------------------------------------------------

    def get_app(self, app_id: str) -> dict[str, Any] | None:
        """Return the cached spec for ``app_id``, or ``None``."""
        return self._apps.get(app_id)

    def get_all_apps(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of the full app cache."""
        return dict(self._apps)

    # ------------------------------------------------------------------
    # Background watcher
    # ------------------------------------------------------------------

    async def watch_updates(self) -> None:
        """Background task: list + watch AppInstance CRs and maintain ``self._apps``."""
        await self._load_config()
        assert self._custom_api is not None

        while True:
            try:
                if self._app_instance_rv == 0:
                    initial_list = await self._custom_api.list_namespaced_custom_object(
                        group=CRD_GROUP,
                        version=CRD_VERSION,
                        namespace=K8S_NAMESPACE,
                        plural=CRD_PLURAL_APP_INSTANCES,
                    )
                    for item in initial_list.get("items", []):
                        self._handle_object(item, "EXISTING")
                    self._app_instance_rv = int(initial_list["metadata"]["resourceVersion"])

                async with Watch().stream(
                    self._custom_api.list_namespaced_custom_object,
                    group=CRD_GROUP,
                    version=CRD_VERSION,
                    namespace=K8S_NAMESPACE,
                    plural=CRD_PLURAL_APP_INSTANCES,
                    watch=True,
                    resource_version=str(self._app_instance_rv),
                    timeout_seconds=300,
                ) as stream:
                    async for event in stream:
                        obj = event["object"]
                        self._handle_object(obj, event.get("type", "UNKNOWN"))
                        self._app_instance_rv = int(obj["metadata"]["resourceVersion"])

            except asyncio.CancelledError:
                raise
            except client.ApiException as e:
                if getattr(e, "status", None) == 410:
                    logger.warning("AppInstance watch expired (410 Gone). Relisting.")
                    self._app_instance_rv = 0
                    self._apps.clear()
                    continue
                logger.error("Error watching AppInstance (API): %s", e)
                await asyncio.sleep(5)
            except Exception as e:
                logger.error("Error watching AppInstance: %s", e)
                await asyncio.sleep(5)

    def _handle_object(self, obj: dict[str, Any], event_type: str) -> None:
        spec = obj.get("spec", {})
        app_id = spec.get("appId")

        if not app_id:
            logger.warning("AppInstance object missing appId, skipping: %s", obj.get("metadata", {}).get("name"))
            return

        if event_type in ("EXISTING", "ADDED", "MODIFIED"):
            self._apps[app_id] = spec
            logger.info("AppRegistry: %s app '%s' (state=%s)", event_type, app_id, spec.get("state"))
        elif event_type == "DELETED":
            self._apps.pop(app_id, None)
            logger.info("AppRegistry: DELETED app '%s'", app_id)

    # ------------------------------------------------------------------
    # Sync barrier
    # ------------------------------------------------------------------

    async def sync_watchers(self) -> None:
        """Wait until the local watcher has caught up to the target app RV."""
        target_app_rv = await self.get_latest_app_rv()

        while self._app_instance_rv < target_app_rv:
            await asyncio.sleep(SYNC_WATCHERS_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Close underlying Kubernetes client resources."""
        if self._api_client:
            await self._api_client.close()
            self._api_client = None
            self._custom_api = None
