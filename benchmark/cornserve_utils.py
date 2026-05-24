"""Utilities for interacting with Cornserve gateway and managing apps and tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import aiohttp
from app_utils import (
    app_to_expected_task_replica_specs,
    app_to_expected_tasks,
    create_app_source,
    extract_tasks_from_source,
)
from schema import AppType, Experiment

from cornserve.services.gateway.models import (
    AppRegistrationRequest,
    DeployedTaskInfo,
    RegistrationErrorResponse,
    RegistrationFinalResponse,
    RegistrationInitialResponse,
    RegistrationStatusEvent,
    ResourceSnapshot,
)
from cornserve.task.base import UnitTask

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
GATEWAY_URL = "http://localhost:30080"
GATEWAY_MASTER_URL = "http://localhost:30081"


def get_gpu_count(experiment: Experiment) -> int:
    """Compute the total number of GPUs required by an experiment.

    Uses resolved unit-task replica specs for all app types.
    """
    total = 0
    for task, num_replicas in app_to_expected_task_replica_specs(experiment.app):
        parallel_size = (
            getattr(task, "tp_size", None) or getattr(task, "sp_size", None) or 1
        )
        total += num_replicas * parallel_size
    return total


async def release(experiment: Experiment) -> None:
    """Release all GPUs owned by the experiment's task(s).

    Inverse of :func:`scale` — queries the current resource snapshot,
    sets ``gpu.owner = None`` for every GPU owned by the experiment's
    task manager(s), updates ``free_gpus`` counts, and applies the snapshot.
    """
    app = experiment.app
    expected_tasks = app_to_expected_tasks(app)
    our_tm_ids: set[str] = set()
    for et in expected_tasks:
        deployed_info = await find_deployed_task(et)
        our_tm_ids.add(deployed_info.task_manager_id)

    # Get current resource snapshot
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        async with session.get(f"{GATEWAY_URL}/resources/snapshot") as response:
            response.raise_for_status()
            snapshot_data = await response.json()

    snapshot = ResourceSnapshot.model_validate(snapshot_data)

    # Release all GPUs owned by our task(s)
    released = 0
    for node in snapshot.nodes:
        for gpu in node.gpus:
            if gpu.owner in our_tm_ids:
                gpu.owner = None
                released += 1

    # Update free_gpus counts
    for node in snapshot.nodes:
        node.free_gpus = sum(1 for gpu in node.gpus if gpu.owner is None)

    print(f"Releasing {released} GPUs from {our_tm_ids}")
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        async with session.post(
            f"{GATEWAY_MASTER_URL}/resources/apply_snapshot",
            json=snapshot.model_dump(mode="json"),
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise ValueError(
                    f"Failed to apply resource snapshot (HTTP {response.status}): {body}"
                )
            result = await response.json()
            print(f"Release snapshot applied: {result}")


async def clear_all() -> None:
    """Unregister all apps, tear down all tasks, and release all GPUs.

    1. Unregisters every app.
    2. Tears down all deployed task managers via ``DELETE /tasks/usage``.
    3. Applies a snapshot with all GPU owners set to ``None``.
    """
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        # 1. Unregister all apps
        async with session.get(f"{GATEWAY_URL}/app/list") as response:
            response.raise_for_status()
            app_states = await response.json()

        for app_id in list(app_states):
            async with session.post(
                f"{GATEWAY_MASTER_URL}/app/unregister/{app_id}"
            ) as response:
                if response.status == 200:
                    print(f"Unregistered app: {app_id}")
                elif response.status != 404:
                    body = await response.text()
                    print(f"Warning: Failed to unregister {app_id}: {body}")

        # 2. Tear down all deployed tasks so their task managers release GPUs
        async with session.get(f"{GATEWAY_URL}/tasks/list") as response:
            response.raise_for_status()
            deployed_list = await response.json()

        if deployed_list:
            deployed_infos = [DeployedTaskInfo.model_validate(d) for d in deployed_list]
            # Build the _task_list format expected by UnitTaskList deserialization
            task_list_items: list[dict] = []
            for info in deployed_infos:
                task_list_items.append(
                    {
                        "class_name": info.task_class,
                        "task": json.dumps(info.task_definition),
                    }
                )
                print(
                    f"Tearing down task: {info.task_class} (tm={info.task_manager_id})"
                )

            async with session.request(
                "DELETE",
                f"{GATEWAY_MASTER_URL}/tasks/usage",
                json={"_task_list": task_list_items},
            ) as response:
                if response.status == 200:
                    print(f"Torn down {len(task_list_items)} task(s)")
                else:
                    body = await response.text()
                    print(
                        f"Warning: Failed to tear down tasks (HTTP {response.status}): {body}"
                    )

            # Give task managers time to fully shut down
            await asyncio.sleep(2)

        # 3. Clear all GPU owners
        async with session.get(f"{GATEWAY_URL}/resources/snapshot") as response:
            response.raise_for_status()
            snapshot_data = await response.json()

    snapshot = ResourceSnapshot.model_validate(snapshot_data)
    cleared = 0
    for node in snapshot.nodes:
        for gpu in node.gpus:
            if gpu.owner is not None:
                gpu.owner = None
                cleared += 1
        node.free_gpus = len(node.gpus)

    if cleared > 0:
        async with aiohttp.ClientSession(
            trust_env=True, timeout=AIOHTTP_TIMEOUT
        ) as session:
            async with session.post(
                f"{GATEWAY_MASTER_URL}/resources/apply_snapshot",
                json=snapshot.model_dump(mode="json"),
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise ValueError(
                        f"Failed to clear snapshot (HTTP {response.status}): {body}"
                    )
        print(f"Cleared {cleared} GPU assignments")
    else:
        print("No GPU assignments to clear")


async def unregister_app(app_id: str) -> None:
    """Unregister an app from the CornServe gateway.

    Args:
        app_id: The registered app ID to unregister.
    """
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        async with session.post(f"{GATEWAY_MASTER_URL}/app/unregister/{app_id}") as response:
            if response.status == 404:
                print(f"Warning: App {app_id} not found during unregister.")
                return
            if response.status != 200:
                body = await response.text()
                raise ValueError(
                    f"Failed to unregister app {app_id} (HTTP {response.status}): {body}"
                )
    print(f"Unregistered app: {app_id}")


def _tasks_match(task1: UnitTask, task2: UnitTask) -> bool:
    """Check if two tasks match by class name and field values.

    This is a lightweight alternative to ``is_equivalent_to`` that does not
    require execution descriptors to be registered.
    """
    if task1.__class__.__name__ != task2.__class__.__name__:
        return False
    root_cls = task1.root_unit_task_cls
    for field_name in root_cls.model_fields:
        if field_name == "id":
            continue
        if getattr(task1, field_name, None) != getattr(task2, field_name, None):
            return False
    return True


async def register_app(app: AppType, verify: bool = True) -> str:
    """Register an app with the CornServe gateway using templates.

    Args:
        app: The app configuration
        verify: If True, verify that the generated source code produces the expected task
                using is_equivalent_to() before registering

    Returns:
        The registered app ID

    Raises:
        ValueError: If verify=True and the generated task doesn't match the expected task
    """
    # Generate source code from template
    source_code = create_app_source(app)

    # Optionally verify that the source code produces the expected task(s)
    if verify:
        expected_tasks = app_to_expected_tasks(app)
        actual_tasks = extract_tasks_from_source(source_code)

        if not actual_tasks:
            raise ValueError("Generated source code does not contain any tasks")

        # Check that every expected task has a match in the actual tasks.
        # Uses field-by-field comparison instead of is_equivalent_to() to
        # avoid requiring execution descriptors to be registered.
        for expected_task in expected_tasks:
            if not any(
                _tasks_match(actual_task, expected_task) for actual_task in actual_tasks
            ):
                raise ValueError(
                    f"Generated task does not match expected task.\n"
                    f"Expected: {expected_task.__class__.__name__} {expected_task.model_dump(exclude={'id'})}\n"
                    f"Actual: {[t.__class__.__name__ for t in actual_tasks]}"
                )

    # Register with gateway (SSE streaming response)
    request = AppRegistrationRequest(source_code=source_code)
    app_id = None
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        async with session.post(
            f"{GATEWAY_MASTER_URL}/app/register",
            json=request.model_dump(),
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.content.iter_any():
                buffer += chunk.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    event = RegistrationStatusEvent.model_validate_json(line[6:]).event
                    if isinstance(event, RegistrationInitialResponse):
                        if app_id is not None:
                            raise RuntimeError(
                                "Received more than one initial responses."
                            )
                        app_id = event.app_id
                    if isinstance(event, RegistrationErrorResponse):
                        raise RuntimeError(f"Registration failed: {event.message}")
                    if isinstance(event, RegistrationFinalResponse):
                        break

    if not app_id:
        raise RuntimeError("No app ID received during registration.")
    return app_id


async def check_apps(app_ids: list[str]) -> None:
    """Check if the specified apps are running."""
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        try:
            async with session.get(f"{GATEWAY_URL}/app/list") as response:
                response.raise_for_status()
                app_states = await response.json()
                for app_id in app_ids:
                    if app_id not in app_states:
                        raise ValueError(f"App {app_id} is not registered.")
                    if app_states[app_id] != "ready":
                        raise ValueError(f"App {app_id} is not running.")
        except aiohttp.ClientError as e:
            raise ValueError("Failed to connect to the gateway.") from e


async def check_tasks_ready(app: "AppType") -> None:
    """Check that every unit task for *app* has finished deploying.

    The app-level state (checked by ``check_apps``) turns ``"ready"`` before
    the underlying vLLM pods finish loading the model.  This function checks
    the task-level state for only the tasks belonging to *app*, so it is safe
    to call concurrently with other batched experiments that may still be
    deploying their own tasks.

    Raises ``ValueError`` if any expected task is not yet in ``"ready"`` state.
    ``find_deployed_task`` already filters by state == "ready", so it raises
    ``ValueError`` when a matching task exists but is still deploying.
    """
    from app_utils import app_to_expected_tasks

    for expected_task in app_to_expected_tasks(app):
        await find_deployed_task(expected_task)


async def find_deployed_task(expected_task: UnitTask) -> DeployedTaskInfo:
    """Find a deployed task that is equivalent to the expected task.

    Compares task class name and field values directly from the task_definition dict.

    Args:
        expected_task: The expected task definition

    Returns:
        The deployed task info

    Raises:
        ValueError: If no matching deployed task is found
    """
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        async with session.get(f"{GATEWAY_URL}/tasks/list") as response:
            response.raise_for_status()
            deployed_list = await response.json()

    deployed_infos = [DeployedTaskInfo.model_validate(d) for d in deployed_list]

    for deployed_info in deployed_infos:
        if deployed_info.state != "ready":
            continue
        if deployed_info.task_class != expected_task.__class__.__name__:
            continue
        deployed_task = expected_task.__class__.model_validate(
            deployed_info.task_definition
        )
        if _tasks_match(expected_task, deployed_task):
            return deployed_info

    raise ValueError(
        f"No deployed task found equivalent to {expected_task.__class__.__name__} "
        f"with config {expected_task.model_dump(exclude={'id'})}"
    )


def _place_replicas(
    snapshot: ResourceSnapshot,
    tm_id: str,
    num_replicas: int,
    tp_size: int,
) -> None:
    """Assign GPUs to *tm_id* in *snapshot* (mutates in place).

    Finds contiguous groups of *tp_size* free GPUs on each node and
    assigns them to the task manager until *num_replicas* are placed.

    Raises:
        ValueError: If not enough adjacent free GPUs are available.
    """
    placed = 0
    for node in snapshot.nodes:
        if placed >= num_replicas:
            break
        free_gpus = sorted(
            [gpu for gpu in node.gpus if gpu.owner is None],
            key=lambda g: g.local_rank,
        )
        i = 0
        while i + tp_size <= len(free_gpus) and placed < num_replicas:
            group = free_gpus[i : i + tp_size]
            ranks = [g.local_rank for g in group]
            if ranks == list(range(ranks[0], ranks[0] + tp_size)):
                for gpu in group:
                    gpu.owner = tm_id
                placed += 1
                i += tp_size
            else:
                i += 1

    if placed < num_replicas:
        raise ValueError(
            f"Cannot place {num_replicas} replica(s) of '{tm_id}' "
            f"with tp_size={tp_size}: only placed {placed}. "
            f"Not enough adjacent free GPUs on any node."
        )


async def scale(experiment: Experiment) -> None:
    """Scale tasks based on the provided experiment configuration using snapshot API.

    Args:
        experiment: The experiment configuration containing app and replica info
    """
    app = experiment.app

    # 1. Find deployed task(s) and desired replicas for each unit task.
    task_replica_specs = app_to_expected_task_replica_specs(app)
    if not task_replica_specs:
        raise ValueError(
            f"No task replica specs found for app type {type(app).__name__}"
        )

    deployed_specs: list[tuple[str, int, int, str]] = []
    for expected_task, num_replicas in task_replica_specs:
        deployed_info = await find_deployed_task(expected_task)
        parallel_size = (
            getattr(expected_task, "tp_size", None)
            or getattr(expected_task, "sp_size", None)
            or 1
        )
        deployed_specs.append(
            (
                deployed_info.task_manager_id,
                num_replicas,
                parallel_size,
                getattr(expected_task, "make_name")(),
            )
        )

    tm_ids = [tm_id for tm_id, _, _, _ in deployed_specs]

    # 2. Get current resource snapshot
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        async with session.get(f"{GATEWAY_URL}/resources/snapshot") as response:
            response.raise_for_status()
            snapshot_data = await response.json()

    snapshot = ResourceSnapshot.model_validate(snapshot_data)

    # 3. Release all GPUs currently owned by our task(s)
    our_tm_ids = set(tm_ids)
    for node in snapshot.nodes:
        for gpu in node.gpus:
            if gpu.owner in our_tm_ids:
                gpu.owner = None

    # 4. Place replicas for each unit task.
    placement_plan: list[str] = []
    for tm_id, num_replicas, parallel_size, task_name in deployed_specs:
        _place_replicas(snapshot, tm_id, num_replicas, parallel_size)
        placement_plan.append(
            f"{task_name} ({tm_id}) -> {num_replicas} x {parallel_size} GPUs"
        )

    print("Applying resource snapshot: " + "; ".join(placement_plan))

    # Update free_gpus counts on nodes
    for node in snapshot.nodes:
        node.free_gpus = sum(1 for gpu in node.gpus if gpu.owner is None)

    # 5. Apply the new snapshot
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        async with session.post(
            f"{GATEWAY_MASTER_URL}/resources/apply_snapshot",
            json=snapshot.model_dump(mode="json"),
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise ValueError(
                    f"Failed to apply resource snapshot (HTTP {response.status}): {body}"
                )
            result = await response.json()
            print(f"Snapshot applied: {result}")


logger = logging.getLogger(__name__)

CORNSERVE_NAMESPACE = "cornserve"


class ExecutorLogStreamer:
    """Stream task-executor pod logs to files in the background.

    Usage::

        streamer = await ExecutorLogStreamer.start(output_dir)
        try:
            ...  # run benchmark
        finally:
            await streamer.stop()

    Logs are written continuously via ``kubectl logs -f``, so they are
    preserved even if the executor pod crashes mid-benchmark.
    """

    def __init__(self) -> None:
        self._procs: list[tuple[str, asyncio.subprocess.Process, Path]] = []

    @classmethod
    async def start(
        cls,
        output_dir: Path,
        pod_name_prefix: str | None = None,
        pod_name_prefixes: list[str] | None = None,
    ) -> "ExecutorLogStreamer":
        """Find executor pods and start ``kubectl logs -f`` for each.

        Args:
            output_dir: Directory to save log files.
            pod_name_prefix: If set, only stream logs from pods whose name
                starts with this prefix.  Mutually exclusive with
                *pod_name_prefixes*.
            pod_name_prefixes: If set, stream logs from pods matching
                **any** prefix in the list.
        """
        prefixes: list[str] | None = pod_name_prefixes
        if prefixes is None and pod_name_prefix is not None:
            prefixes = [pod_name_prefix]

        self = cls()
        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl",
                "get",
                "pods",
                "-n",
                CORNSERVE_NAMESPACE,
                "-l",
                "app=task-executor",
                "-o",
                "jsonpath={.items[*].metadata.name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning("kubectl get pods failed: %s", stderr.decode())
                return self

            pod_names = stdout.decode().split()
            if prefixes:
                pod_names = [
                    p for p in pod_names if any(p.startswith(pfx) for pfx in prefixes)
                ]
            if not pod_names:
                logger.info("No task-executor pods found for log streaming")
                return self

            logs_dir = output_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            for pod_name in pod_names:
                log_path = logs_dir / f"{pod_name}.log"
                log_file = open(log_path, "wb")  # noqa: SIM115
                follow_proc = await asyncio.create_subprocess_exec(
                    "kubectl",
                    "logs",
                    "-f",
                    pod_name,
                    "-n",
                    CORNSERVE_NAMESPACE,
                    stdout=log_file,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                self._procs.append((pod_name, follow_proc, log_path))
                logger.info("Streaming logs from %s to %s", pod_name, log_path)

        except Exception:
            logger.exception("Failed to start executor log streaming")

        return self

    async def stop(self) -> list[Path]:
        """Terminate all ``kubectl logs -f`` processes and return saved paths."""
        saved: list[Path] = []
        for pod_name, proc, log_path in self._procs:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            # Close the stdout file handle
            if proc.stdout and not proc.stdout.at_eof():
                proc.stdout.feed_eof()
            if log_path.exists() and log_path.stat().st_size > 0:
                saved.append(log_path)
                logger.info(
                    "Saved %d bytes of logs from %s", log_path.stat().st_size, pod_name
                )
        self._procs.clear()
        return saved
