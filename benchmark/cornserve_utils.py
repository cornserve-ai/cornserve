"""Utilities for interacting with Cornserve gateway and managing apps and tasks."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import aiohttp
import requests
from app_utils import create_eric_app, create_mllm_app
from schema import (
    CornserveConfig,
    EPDConfig,
    EricConfig,
    ExperimentConfig,
    PDConfig,
    VLLMConfig,
)

from cornserve.services.gateway.models import (
    AppRegistrationRequest,
    RegistrationErrorResponse,
    RegistrationFinalResponse,
    RegistrationInitialResponse,
    RegistrationStatusEvent,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
GATEWAY_URL = "http://localhost:30080"


def register_app(
    model_id: str,
    app_type: Literal["ev", "v", "e", "epd", "pd"],
) -> str:
    """Register an app with the CornServe gateway, and return the app ID."""
    if app_type == "ev":
        source_code = create_mllm_app(
            model_id=model_id,
            task_class="MLLMTask",
            encoder_fission=True,
        )
    elif app_type == "v":
        source_code = create_mllm_app(
            model_id=model_id,
            task_class="MLLMTask",
            encoder_fission=False,
        )
    elif app_type == "e":
        source_code = create_eric_app(
            model_id=model_id,
        )
    elif app_type == "epd":
        source_code = create_mllm_app(
            model_id=model_id,
            task_class="DisaggregatedMLLMTask",
            encoder_fission=True,
        )
    elif app_type == "pd":
        source_code = create_mllm_app(
            model_id=model_id,
            task_class="DisaggregatedMLLMTask",
            encoder_fission=False,
        )
    else:
        raise NotImplementedError(f"Unsupported app_type: {app_type}.")

    request = AppRegistrationRequest(source_code=source_code)
    response = requests.post(
        f"{GATEWAY_URL}/app/register",
        json=request.model_dump(),
        timeout=(5, 1200),
        stream=True,
    )
    response.raise_for_status()
    response_iter = response.iter_lines(decode_unicode=True)

    app_id = None
    # Get immediate initial response
    for line in response_iter:
        if not line or not line.startswith("data: "):
            continue
        event = RegistrationStatusEvent.model_validate_json(line[6:]).event
        if isinstance(event, RegistrationInitialResponse):
            if app_id is not None:
                raise RuntimeError("Received more than one initial responses.")
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
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        for app_id in app_ids:
            try:
                async with session.get("http://127.0.0.1:30080/app/list") as response:
                    response.raise_for_status()
                    app_states = await response.json()
                    for app_id in app_ids:
                        if app_id not in app_states:
                            raise ValueError(f"App {app_id} is not registered.")
                        if app_states[app_id] != "ready":
                            raise ValueError(f"App {app_id} is not running.")
            except aiohttp.ClientError as e:
                raise ValueError("Failed to connect to the gateway.") from e


async def clear_task_executors() -> None:
    """Clear all task executors in Cornserve."""
    print("Clearing all task executors...")
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.get("http://127.0.0.1:30080/tasks/list") as response:
                response.raise_for_status()
                task_states = await response.json()
                task_ids = [state[1] for state in task_states]
        except aiohttp.ClientError as e:
            raise ValueError("Failed to clear task executors.") from e

        async def scale_to_zero(task_id: str) -> None:
            print(f"Scaling task {task_id} to zero replicas...")
            payload = {"task_id": task_id, "num_gpus": -1}
            while True:
                async with session.post("http://127.0.0.1:30080/task/scale", json=payload) as response:
                    if response.status == 500:
                        # TODO:
                        # migrate 403 from jm-benchmark branch
                        break
                    if response.status != 200:
                        raise ValueError(f"Unexpecged error while scaling task {task_id} to zero: {response}")
            print(f"Task {task_id} scaled to zero replicas.")

        coros = [scale_to_zero(task_id) for task_id in task_ids]
        await asyncio.gather(*coros)


async def get_tasks() -> list[tuple[dict[str, Any], str, str]]:
    """Get all tasks in Cornserve."""
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.get("http://127.0.0.1:30080/tasks/list") as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise ValueError("Failed to get task IDs.") from e


async def scale_task_with_num_gpus(task_id: str, num_gpus: int) -> None:
    """Scale a task to the specified number of GPUs."""
    print(f"Scaling task {task_id} with {num_gpus} gpus...")
    scale_endpoint = "http://127.0.0.1:30080/task/scale"
    payload = {"task_id": task_id, "num_gpus": num_gpus}
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(scale_endpoint, json=payload) as response:
                if response.status == 200:
                    print(f"Task {task_id} scaled to {num_gpus} replicas.")
                else:
                    raise ValueError(f"Failed to scale task {task_id}: {response}")
        except aiohttp.ClientError as e:
            raise ValueError("Failed to scale task.") from e


async def scale(config: ExperimentConfig) -> None:
    """Scale tasks based on the provided experiment configuration."""
    await clear_task_executors()
    tasks = await get_tasks()
    model_id = config.model_id
    if isinstance(config.backend_config, EPDConfig):
        for task_def, task_id, state in tasks:
            if state != "ready":
                continue
            if "encodertask" in task_id and model_id in task_def["model_ids"]:
                encoder_task_id = task_id
            elif "prefillllmunittask" in task_id and model_id == task_def["model_id"] \
                    and task_def["receive_embeddings"] == True:
                prefill_task_id = task_id
            elif "decodellmunittask" in task_id and model_id == task_def["model_id"] \
                    and task_def["receive_embeddings"] == True:
                decode_task_id = task_id
        assert all([prefill_task_id, decode_task_id, encoder_task_id]), (
            "Not all tasks are running. Please check the task and app states."
        )
        await scale_task_with_num_gpus(
            task_id=prefill_task_id,
            num_gpus=config.backend_config.num_prefills * config.backend_config.prefill_tp_size,
        )
        await scale_task_with_num_gpus(
            task_id=decode_task_id,
            num_gpus=config.backend_config.num_decodes * config.backend_config.decode_tp_size,
        )
        await scale_task_with_num_gpus(
            task_id=encoder_task_id,
            num_gpus=config.backend_config.num_erics * config.backend_config.eric_tp_size,
        )
    elif isinstance(config.backend_config, PDConfig):
        for task_def, task_id, state in tasks:
            if state != "ready":
                continue
            if "prefillllmunittask" in task_id and model_id == task_def["model_id"] \
                and task_def["receive_embeddings"] == False:
                prefill_task_id = task_id
            elif "decodellmunittask" in task_id and model_id == task_def["model_id"] \
                and task_def["receive_embeddings"] == False:
                decode_task_id = task_id
        assert all([prefill_task_id, decode_task_id]), (
            "Not all tasks are running. Please check the task and app states."
        )
        await scale_task_with_num_gpus(
            task_id=prefill_task_id,
            num_gpus=config.backend_config.num_prefills * config.backend_config.prefill_tp_size,
        )
        await scale_task_with_num_gpus(
            task_id=decode_task_id,
            num_gpus=config.backend_config.num_decodes * config.backend_config.decode_tp_size,
        )
    elif isinstance(config.backend_config, VLLMConfig):
        for task_def, task_id, state in tasks:
            if state != "ready":
                continue
            if "llmunittask" in task_id and model_id == task_def["model_id"] \
                    and task_def["receive_embeddings"] == False:
                vllm_task_id = task_id
        assert vllm_task_id, "No vLLM task found. Please check the task and app states."
        await scale_task_with_num_gpus(
            task_id=vllm_task_id,
            num_gpus=config.backend_config.num_replicas * config.backend_config.tp_size,
        )
    elif isinstance(config.backend_config, CornserveConfig):
        for task_def, task_id, state in tasks:
            if state != "ready":
                continue
            if "encodertask" in task_id and model_id in task_def["model_ids"] \
                    and "dummyencodertask" not in task_id:
                eric_task_id = task_id
            elif "llmunittask" in task_id and model_id == task_def["model_id"] \
                    and task_def["receive_embeddings"] == True:
                vllm_task_id = task_id
        assert all([eric_task_id, vllm_task_id]), "Not all tasks are running. Please check the task and app states."
        await scale_task_with_num_gpus(
            task_id=vllm_task_id,
            num_gpus=config.backend_config.num_vllms * config.backend_config.vllm_tp_size,
        )
        await scale_task_with_num_gpus(
            task_id=eric_task_id,
            num_gpus=config.backend_config.num_erics * config.backend_config.eric_tp_size,
        )
    elif isinstance(config.backend_config, EricConfig):
        for task_def, task_id, state in tasks:
            if state != "ready":
                continue
            if "dummyencodertask" in task_id and model_id in task_def["model_ids"]:
                eric_task_id = task_id
        assert eric_task_id, "No Eric task found. Please check the task and app states."
        await scale_task_with_num_gpus(
            task_id=eric_task_id,
            num_gpus=config.backend_config.num_replicas * config.backend_config.tp_size,
        )
    else:
        raise NotImplementedError(f"Backend config {config.backend_config} is not supported.")


if __name__ == "__main__":
    app_id = register_app(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        app_type="ev",
    )
    print(f"Registered app with ID: {app_id}")
    ev_id = register_app("Qwen/Qwen2.5-VL-32B-Instruct", "ev")
    experiment_config = ExperimentConfig(
        backend_config=CornserveConfig(
            num_vllms=2,
            vllm_tp_size=2,
            num_erics=4,
            eric_tp_size=1,
        ),
        app_id=ev_id,
        model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        request_rate=5,
        input_len=64,
        output_len=64,
        image_count=1,
        num_prompts=1000,
        image_width=224,
        image_height=224,
    )
    asyncio.run(scale(experiment_config))
