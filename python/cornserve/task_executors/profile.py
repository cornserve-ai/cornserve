"""UnitTask profiling system for GPU resource allocation.

This module provides a profile-based system for determining GPU resource requirements
for UnitTask instances. Profiles are stored as Kubernetes CRs for durability.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kubernetes_asyncio import client, config
from pydantic import BaseModel

from cornserve.constants import (
    CRD_GROUP,
    CRD_KIND_UNIT_TASK_PROFILE,
    CRD_PLURAL_UNIT_TASK_PROFILES,
    CRD_VERSION,
    K8S_NAMESPACE,
)
from cornserve.logging import get_logger
from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY

if TYPE_CHECKING:
    from cornserve.task.base import UnitTask

logger = get_logger(__name__)


class ProfileInfo(BaseModel):
    """Profile information for a specific GPU count.

    This class will eventually hold performance-related metadata as well.

    Attributes:
        launch_args: Additional arguments to pass when launching the task
    """

    launch_args: list[str] = []


@dataclass
class UnitTaskProfile:
    """Profile mapping GPU counts to profile information for a UnitTask.

    Attributes:
        task: The UnitTask instance this profile applies to
        num_gpus_to_profile: Mapping from GPU count to ProfileInfo
    """

    task: UnitTask
    num_gpus_to_profile: dict[int, ProfileInfo]

    @classmethod
    def from_json_file(cls, file_path: Path) -> UnitTaskProfile:
        """Load a UnitTaskProfile from a JSON file.

        Args:
            file_path: Path to the JSON file containing the profile

        Returns:
            UnitTaskProfile instance

        Raises:
            ValueError: If the file format is invalid
            FileNotFoundError: If the file doesn't exist
        """
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Parse the task from the JSON data using the global registry
            task_class_name = data["task"]["__class__"]
            task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task(task_class_name)
            task = task_cls.model_validate_json(json.dumps(data["task"]))

            # Parse GPU profile information
            num_gpus_to_profile: dict[int, ProfileInfo] = {}
            for gpu_count_str, profile_data in data["num_gpus_to_profile"].items():
                gpu_count = int(gpu_count_str)
                num_gpus_to_profile[gpu_count] = ProfileInfo(**profile_data)

            return cls(task=task, num_gpus_to_profile=num_gpus_to_profile)

        except (KeyError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid profile file format in {file_path}: {e}") from e

    def to_json_file(self, file_path: Path) -> None:
        """Save the UnitTaskProfile to a JSON file.

        Args:
            file_path: Path where to save the JSON file
        """
        # Prepare data for JSON serialization
        task_data = json.loads(self.task.model_dump_json())
        task_data["__class__"] = self.task.__class__.__name__

        data = {
            "task": task_data,
            "num_gpus_to_profile": {
                str(gpu_count): profile_info.model_dump()
                for gpu_count, profile_info in self.num_gpus_to_profile.items()
            },
        }

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


def _sanitize_k8s_name(name: str) -> str:
    """Convert a string to a valid K8s resource name."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name)  # Collapse multiple dashes
    name = name.strip("-")
    # Must be <= 63 chars and start with alphanumeric
    if name and not name[0].isalnum():
        name = "p" + name
    return name[:63]


def _generate_profile_name(task_dict: dict[str, Any]) -> str:
    """Generate a unique, deterministic name for a profile CR."""
    task_class = task_dict.get("__class__", "unknown")
    model_id = task_dict.get("model_id", "")

    # Create hash for uniqueness
    content_hash = hashlib.sha256(json.dumps(task_dict, sort_keys=True).encode()).hexdigest()[:8]

    # Build name
    base_name = f"{task_class}-{model_id}".replace("/", "-") if model_id else task_class
    return _sanitize_k8s_name(f"profile-{base_name}-{content_hash}")


class UnitTaskProfileManager:
    """Manager for UnitTask profiles with CRD-based storage.

    This manager handles loading profiles from Kubernetes CRs and provides
    async CRUD operations for profile resources.
    """

    def __init__(self) -> None:
        """Initialize lazy Kubernetes API clients."""
        self._api_client: client.ApiClient | None = None
        self._custom_api: client.CustomObjectsApi | None = None

    async def _load_config(self) -> None:
        """Load K8s config and initialize API clients."""
        if self._api_client:
            return
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config for UnitTaskProfileManager.")
        except config.ConfigException as e:
            logger.error("Failed to load Kubernetes config: %s", e)
            raise RuntimeError("Could not load Kubernetes configuration") from e

        self._api_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(self._api_client)

    async def create_profile(
        self,
        task_dict: dict[str, Any],
        num_gpus_to_profile: dict[str, dict[str, Any]],
        namespace: str = K8S_NAMESPACE,
    ) -> str:
        """Create or update a UnitTaskProfile CR."""
        await self._load_config()
        assert self._custom_api is not None

        profile_name = _generate_profile_name(task_dict)

        body = {
            "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
            "kind": CRD_KIND_UNIT_TASK_PROFILE,
            "metadata": {"name": profile_name, "namespace": namespace},
            "spec": {
                "task": task_dict,
                "numGpusToProfile": num_gpus_to_profile,
            },
        }

        try:
            await self._custom_api.create_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural=CRD_PLURAL_UNIT_TASK_PROFILES,
                body=body,
            )
            logger.info("Created unit task profile CR: %s", profile_name)
            return profile_name
        except client.ApiException as e:
            if e.status == 409:
                # Profile already exists - replace it
                await self._custom_api.replace_namespaced_custom_object(
                    group=CRD_GROUP,
                    version=CRD_VERSION,
                    namespace=namespace,
                    plural=CRD_PLURAL_UNIT_TASK_PROFILES,
                    name=profile_name,
                    body=body,
                )
                logger.info("Updated unit task profile CR: %s", profile_name)
                return profile_name
            raise

    async def get_profile(self, task: UnitTask) -> UnitTaskProfile:
        """Get the profile for a UnitTask by querying CRs.

        Returns a default profile (1 GPU) if no matching profile is found.

        Args:
            task: The UnitTask to get the profile for
        """
        await self._load_config()
        assert self._custom_api is not None

        try:
            resp = await self._custom_api.list_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=K8S_NAMESPACE,
                plural=CRD_PLURAL_UNIT_TASK_PROFILES,
            )

            for item in resp.get("items", []):
                spec = item.get("spec", {})
                task_data = spec.get("task", {})

                task_class_name = task_data.get("__class__")
                if not task_class_name:
                    continue

                try:
                    task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task(task_class_name)
                    profile_task = task_cls.model_validate(task_data)

                    if profile_task.is_equivalent_to(task):
                        num_gpus_to_profile: dict[int, ProfileInfo] = {}
                        for gpu_count_str, profile_data in spec.get("numGpusToProfile", {}).items():
                            gpu_count = int(gpu_count_str)
                            num_gpus_to_profile[gpu_count] = ProfileInfo(**profile_data)

                        if not num_gpus_to_profile:
                            raise ValueError(f"Profile for task {task} has no GPU profiles defined")

                        logger.info("Found profile for task %s", task)
                        return UnitTaskProfile(task=profile_task, num_gpus_to_profile=num_gpus_to_profile)
                except Exception as e:
                    logger.warning("Failed to parse profile CR %s: %s", item.get("metadata", {}).get("name"), e)
                    continue

        except client.ApiException as e:
            logger.error("Failed to list unit task profiles: %s", e)

        # No profile found, return default
        logger.info("No profile found for task %s, using default (1 GPU)", task)
        return self.get_default_profile(task)

    def get_default_profile(self, task: UnitTask) -> UnitTaskProfile:
        """Get the default profile for tasks without specific profiles.

        Returns:
            Default profile that can only run with 1 GPU
        """
        return UnitTaskProfile(task=task, num_gpus_to_profile={1: ProfileInfo()})

    async def delete_all_profiles(self, namespace: str = K8S_NAMESPACE) -> None:
        """Delete all UnitTaskProfile CRs."""
        await self._load_config()
        assert self._custom_api is not None

        try:
            resp = await self._custom_api.list_namespaced_custom_object(
                group=CRD_GROUP,
                version=CRD_VERSION,
                namespace=namespace,
                plural=CRD_PLURAL_UNIT_TASK_PROFILES,
            )

            for item in resp.get("items", []):
                name = item.get("metadata", {}).get("name")
                if name:
                    try:
                        await self._custom_api.delete_namespaced_custom_object(
                            group=CRD_GROUP,
                            version=CRD_VERSION,
                            namespace=namespace,
                            plural=CRD_PLURAL_UNIT_TASK_PROFILES,
                            name=name,
                        )
                    except client.ApiException as e:
                        if e.status != 404:
                            logger.error("Failed to delete profile %s: %s", name, e)
                            raise

            logger.info("Deleted all unit task profile CRs.")
        except client.ApiException as e:
            logger.exception("Failed to list profiles for deletion: %s", e)
            raise

    async def shutdown(self) -> None:
        """Close underlying Kubernetes client resources."""
        if self._api_client:
            await self._api_client.close()
            self._api_client = None
            self._custom_api = None
