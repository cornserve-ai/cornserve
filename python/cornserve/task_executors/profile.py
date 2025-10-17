"""UnitTask profiling system for GPU resource allocation.

This module provides a profile-based system for determining GPU resource requirements
for UnitTask instances. Profiles are stored as JSON files and can be deployed to
Kubernetes as ConfigMaps for dynamic resource allocation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from cornserve.constants import UNIT_TASK_PROFILES_DIR
from cornserve.logging import get_logger
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
            # with open(file_path) as f:
            #     data = json.load(f)

            # # Parse the task from the JSON data using task registry
            # task_class_name = data["task"]["__class__"]
            # task_cls, _, _ = TASK_REGISTRY.get(task_class_name)
            # task = task_cls.model_validate_json(json.dumps(data["task"]))

            # # Parse GPU profile information
            # num_gpus_to_profile = {}
            # for gpu_count_str, profile_data in data["num_gpus_to_profile"].items():
            #     gpu_count = int(gpu_count_str)
            #     num_gpus_to_profile[gpu_count] = ProfileInfo(**profile_data)

            # return cls(task=task, num_gpus_to_profile=num_gpus_to_profile)

            # TODO: Temperorily disabled before integrated with CRD
            # The task class can be loaded from the Resource Manager's task registry,
            # and the profile can be loaded from a dedicated CRD in the future.
            return cls(task=None, num_gpus_to_profile={})

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


class UnitTaskProfileManager:
    """Manager for UnitTask profiles with directory-based lookup.

    This manager handles loading profiles from the filesystem by iterating through
    the profile directory each time to ensure fresh data from ConfigMap updates.

    Attributes:
        profile_dir: Directory containing profile JSON files
    """

    def __init__(self, profile_dir: str | Path = UNIT_TASK_PROFILES_DIR) -> None:
        """Initialize the profile manager.

        Args:
            profile_dir: Directory containing profile JSON files
        """
        self.profile_dir = Path(profile_dir)

    def _load_profile_from_file(self, file_path: Path) -> UnitTaskProfile | None:
        """Load a single profile from a file.

        Args:
            file_path: Path to the profile JSON file

        Returns:
            UnitTaskProfile if successfully loaded, None if failed
        """
        try:
            profile = UnitTaskProfile.from_json_file(file_path)
            logger.info("Loaded profile from %s for task %s", file_path, profile.task)
            return profile
        except Exception as e:
            logger.warning("Failed to load profile from %s: %s", file_path, e)
            return None

    def get_profile(self, task: UnitTask) -> UnitTaskProfile:
        """Get the profile for a UnitTask by scanning the profile directory.

        This method iterates through all JSON files in the profile directory
        to find a profile with a task that is equivalent to the given task.
        Returns a default profile (1 GPU) if no specific profile is found.

        Args:
            task: The UnitTask to get the profile for
        """
        if not self.profile_dir.exists():
            logger.info("Profile directory %s does not exist", self.profile_dir)
            return self.get_default_profile(task)

        # Iterate through all JSON files in the profile directory
        for file_path in self.profile_dir.glob("*.json"):
            profile = self._load_profile_from_file(file_path)
            if profile is not None and profile.task.is_equivalent_to(task):
                logger.info("Found profile for task %s in %s", task, file_path)
                if not profile.num_gpus_to_profile:
                    raise ValueError(f"Profile for task {task} in {file_path} has no GPU profiles defined")
                return profile

        # No profile found, return default profile
        logger.info("No profile found for task %s, using default (1 GPU)", task)
        return self.get_default_profile(task)

    def get_default_profile(self, task: UnitTask) -> UnitTaskProfile:
        """Get the default profile for tasks without specific profiles.

        Returns:
            Default profile that can only run with 1 GPU
        """
        return UnitTaskProfile(task=task, num_gpus_to_profile={1: ProfileInfo()})
