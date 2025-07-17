import asyncio
import contextlib
from dataclasses import asdict
from dataclasses import dataclass
import json
import sys
import time
import traceback
from typing import Any, AsyncGenerator
import warnings

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from benchmark_backend import RequestInput, RequestOutput
from benchmark_dataset import SampleRequest, VisionArenaDataset
from schema import (
    BackendConfig,
    CornserveConfig,
    EPDConfig,
    ExperimentConfig,
    PDConfig,
    vLLMConfig,
)
from utils import create_dummy_image, get_image_data_uris

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s[%(name)s:%(lineno)d] - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
MILLISECONDS_TO_SECONDS_CONVERSION = 1000

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


def calculate_metrics(
    input_requests: list[RequestInput],
    outputs: list[RequestOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float] = [90, 95, 99],
    goodput_config_dict: dict[str, float] | None = None,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)  # type: ignore
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,  # type: ignore
        median_ttft_ms=np.median(ttfts or 0) * 1000,  # type: ignore
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles  # type: ignore
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,  # type: ignore
        std_tpot_ms=np.std(tpots or 0) * 1000,  # type: ignore
        median_tpot_ms=np.median(tpots or 0) * 1000,  # type: ignore
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles  # type: ignore
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,  # type: ignore
        std_itl_ms=np.std(itls or 0) * 1000,  # type: ignore
        median_itl_ms=np.median(itls or 0) * 1000,  # type: ignore
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles  # type: ignore
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,  # type: ignore
        std_e2el_ms=np.std(e2els or 0) * 1000,  # type: ignore
        median_e2el_ms=np.median(e2els or 0) * 1000,  # type: ignore
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles  # type: ignore
        ],
    )

    return metrics, actual_output_lens


async def cornserve_invoke(
    request_input: RequestInput,
    pbar: tqdm | None,
) -> RequestOutput:
    api_url = request_input.url
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        content = [{"type": "text", "text": request_input.prompt}]
        for mm_data in request_input.multi_modal_data:
            content.append(mm_data)
        request_data = {
            "model": request_input.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_completion_tokens": request_input.output_len,
            "stream_options": {
                "include_usage": True,
            },
        }

        # Note: this is not implemented in Cornserve yet
        # if request_input.ignore_eos:
        #     request_data["ignore_eos"] = request_input.ignore_eos

        payload = {"request_data": request_data}

        output = RequestOutput()
        # output.input = request_input
        output.prompt_len = request_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post( url=api_url, json=payload) as response:
                if response.status == 200:
                    # iterate over lines
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8").strip()
                        timestamp = time.perf_counter()
                        data = json.loads(line)
                        if choices := data.get("choices"):
                            content = choices[0]["delta"].get("content")
                            # First token
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft
                            # Decoding phase
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            generated_text += content or ""
                        elif usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens")
                            output.usage = usage
                        most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def get_request(
    input_requests: list[RequestInput],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[RequestInput, None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampleRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

def transform_sampled_requests(
    app_id: str,
    config: ExperimentConfig,
    sampled_requests: list[SampleRequest],
) -> list[RequestInput]:
    """
    Transforms the sampled requests from the dataset into
    RequestInput objects for benchmarking.
    """
    # synthesize image_choices images
    image_filenames = [create_dummy_image(
        width=config.image_width,
        height=config.image_height,
        id=i,
    ) for i in range(config.image_choices)]
    np.random.seed(config.seed)
    # first synthesize image_data
    if config.use_synthesized_data:
        print(f"Synthesizing image data with probability {config.image_probability}.")
        for request in sampled_requests:
            if np.random.rand() < config.image_probability:
                # Synthesize image choices
                chosen_image_filenames = list(np.random.choice(
                    image_filenames,
                    size=config.image_count,
                    replace=False,
                ))
                request.filenames = chosen_image_filenames
                request.image_urls = get_image_data_uris(chosen_image_filenames)

    request_inputs = []
    for request in sampled_requests:
        mm_data_list = []
        if config.use_synthesized_data:
            for image_uri in request.image_urls:
                mm_data_list.append({"type": "image_url", "image_url": {"url": image_uri}})
        else:
            mm_data_list = [request.multi_modal_data]
        request_input = RequestInput(
            url=f"http://localhost:30080/app/invoke/{app_id}",
            model=config.model_id,
            prompt=request.prompt,
            prompt_len=request.prompt_len,
            output_len=config.output_len,
            multi_modal_data=mm_data_list,
            filenames=request.filenames,
        )
        request_inputs.append(request_input)
    return request_inputs

async def benchmark(
    request_inputs: list[RequestInput],
    config: ExperimentConfig,
) -> dict[str, Any]:
    # here we assume the cluster is scaled as needed
    max_concurrency = config.max_concurrency
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()
    async def request_func(request_input: RequestInput, pbar: tqdm) -> RequestOutput:
        async with semaphore:
            return await cornserve_invoke(request_input, pbar)

    # first do warmup
    print("Starting warmup phase...")
    warmup_pbar = tqdm(total=config.num_warmups, desc="Warmup")
    coros = [request_func(request_input, warmup_pbar) for request_input in request_inputs[:config.num_warmups]]
    await asyncio.gather(*coros)

    pbar = tqdm(total=len(request_inputs))
    print(f"Starting benchmark with {len(request_inputs)} requests...")
    distribution = "Poisson process" if config.burstiness == 1.0 else "Gamma distribution"
    print(f"Traffic request rate: {config.request_rate}")
    print(f"Burstiness factor: {config.burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    # Start time for overall benchmark
    benchmark_start_time = time.perf_counter()
    tasks = []
    # Generate requests and create tasks
    async for request in get_request(request_inputs, config.request_rate, config.burstiness):
        task = asyncio.create_task(request_func(request_input=request, pbar=pbar))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    benchmark_end_time = time.perf_counter()
    benchmark_duration = benchmark_end_time - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=request_inputs,
        outputs=results,
        dur_s=benchmark_duration,
        tokenizer=AutoTokenizer.from_pretrained(config.model_id),
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    # if goodput_config_dict:
    #     print(
    #         "{:<40} {:<10.2f}".format(
    #             "Request goodput (req/s):", metrics.request_goodput
    #         )
    #     )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )
    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")
    print("=" * 50)

    output_data = {
        "results": [asdict(output) for output in results],
        "metrics": asdict(metrics),
    }

    config.save(output_data)

    return output_data


async def check_apps(app_ids: list[str]) -> None:
    """ Check if the specified apps are running. """
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        for app_id in app_ids:
            try:
                async with session.get(f"http://localhost:30080/app/list") as response:
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
    """ Clear all task executors in Cornserve. """
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.get("http://localhost:30080/tasks/list") as response:
                response.raise_for_status()
                task_states = await response.json()
                task_ids = [state[1] for state in task_states]
        except aiohttp.ClientError as e:
            raise ValueError("Failed to clear task executors.") from e
        async def scale_to_zero(task_id: str) -> None:
            print(f"Scaling task {task_id} to zero replicas...")
            payload = {"task_id": task_id, "num_gpus": -1}
            while True:
                async with session.post("http://localhost:30080/task/scale", json=payload) as response:
                    if response.status == 403:
                        break
                    if response.status != 200:
                        raise ValueError(f"Unexpecged error while scaling task {task_id} to zero: {response}")
            print(f"Task {task_id} scaled to zero replicas.")
        coros = [scale_to_zero(task_id) for task_id in task_ids]
        await asyncio.gather(*coros)

async def get_task_ids() -> list[str]:
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.get("http://localhost:30080/tasks/list") as response:
                response.raise_for_status()
                task_states = await response.json()
                return [state[1] for state in task_states]
        except aiohttp.ClientError as e:
            raise ValueError("Failed to get task IDs.") from e


async def scale_task_with_num_replicas(task_id: str, num_replicas: int) -> None:
    print(f"Scaling task {task_id} with {num_replicas} replicas...")
    scale_endpoint = "http://localhost:30080/task/scale"
    payload={"task_id": task_id, "num_gpus": num_replicas}
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(scale_endpoint, json=payload) as response:
                if response.status == 200:
                    print(f"Task {task_id} scaled to {num_replicas} replicas.")
                else:
                    raise ValueError(f"Failed to scale task {task_id}: {response}")
        except aiohttp.ClientError as e:
            raise ValueError("Failed to scale task.") from e


async def main() -> None:
    configs: set[ExperimentConfig] = set()
    # epd_configs = EPDConfig.create_backend_configs(num_gpus=12)
    # pd_configs = PDConfig.create_backend_configs(num_gpus=12)

    epd_id = "app-60d95848a1664aa78c086d655e9e19cd"
    pd_id = "app-542b1c143d834f4ca32d80aeeb988575"

    backends: list[BackendConfig] = []
    backends.append(PDConfig(num_prefills=2, num_decodes=2))
    backends.append(PDConfig(num_prefills=3, num_decodes=1))
    backends.append(EPDConfig(num_erics=2, num_prefills=1, num_decodes=1))
    backends.append(EPDConfig(num_erics=1, num_prefills=2, num_decodes=1))

    for backend_config in backends:
        configs.add(ExperimentConfig(
            backend_config=backend_config,
            app_id=epd_id if isinstance(backend_config, EPDConfig) else pd_id,
            num_prompts=200,
            request_rate=10.0,
        ))

    # first group configs by batch
    batched_configs: list[list[ExperimentConfig]] = []
    while len(configs):
        cur = configs.pop()
        batch = [cur]
        batchable = {cfg for cfg in configs if cur.batchable_with(cfg)}
        batch.extend(batchable)
        configs -= batchable
        batched_configs.append(batch)

    for batch in batched_configs:
        await check_apps([cfg.app_id for cfg in batch])
        await clear_task_executors()
        shared_config = batch[0]
        np.random.seed(shared_config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            shared_config.model_id,
            tokenizer_mode = "auto",
            trust_remote_code=True,
        )
        sampled_requests: list[SampleRequest]= VisionArenaDataset(
            dataset_path="lmarena-ai/VisionArena-Chat",
            dataset_subset=None,
            dataset_split="train",
            random_seed=shared_config.seed,
            ).sample(
                num_requests=shared_config.num_prompts,
                tokenizer=tokenizer,
                output_len=shared_config.output_len,
                enforced_prompt_len=shared_config.input_len,
            )

        for cfg in batch:
            task_ids = await get_task_ids()
            if isinstance(cfg.backend_config, EPDConfig):
                for task_id in task_ids:
                    if "encodertask" in task_id:
                        encoder_task_id = task_id
                    elif "prefillllmunittask" in task_id:
                        prefill_task_id = task_id
                    elif "decodellmunittask" in task_id:
                        decode_task_id = task_id
                assert all([prefill_task_id, decode_task_id, encoder_task_id]), (
                    "Not all tasks are running. Please check the task and app states."
                )
                await scale_task_with_num_replicas(
                    task_id=encoder_task_id,
                    num_replicas=cfg.backend_config.num_erics,
                )
                await scale_task_with_num_replicas(
                    task_id=prefill_task_id,
                    num_replicas=cfg.backend_config.num_prefills,
                )
                await scale_task_with_num_replicas(
                    task_id=decode_task_id,
                    num_replicas=cfg.backend_config.num_decodes,
                )
            elif isinstance(cfg.backend_config, PDConfig):
                for task_id in task_ids:
                    if "prefillllmunittask" in task_id:
                        prefill_task_id = task_id
                    elif "decodellmunittask" in task_id:
                        decode_task_id = task_id
                assert all([prefill_task_id, decode_task_id]), (
                    "Not all tasks are running. Please check the task and app states."
                )
                await scale_task_with_num_replicas(
                    task_id=prefill_task_id,
                    num_replicas=cfg.backend_config.num_prefills,
                )
                await scale_task_with_num_replicas(
                    task_id=decode_task_id,
                    num_replicas=cfg.backend_config.num_decodes,
                )
            else:
                raise NotImplementedError(f"Backend config {cfg.backend_config} is not supported.")

            request_inputs = transform_sampled_requests(
                app_id=cfg.app_id,
                config=cfg,
                sampled_requests=sampled_requests,
            )
            print(f"Running benchmark for {cfg.backend_config} with {len(request_inputs)} requests...")
            await benchmark(
                request_inputs=request_inputs,
                config=cfg,
            )

            await clear_task_executors()

        print(batch)
        print("=" * 50)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
