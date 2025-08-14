import asyncio
import contextlib
from dataclasses import asdict, dataclass, field, replace
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

from benchmark_dataset import SampleRequest
from image_utils import create_dummy_image, get_image_data_uris
from schema import EricConfig, ExperimentConfig

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
MILLISECONDS_TO_SECONDS_CONVERSION = 1000

@dataclass
class RequestInput:
    """Input for the benchmark request."""
    url: str
    model: str
    prompt: str | Any
    prompt_len: int
    output_len: int
    multi_modal_data: list[dict[str, Any]]
    filenames: list[str] = field(default_factory=list)
    ignore_eos: bool = True

    encoder_fission: bool = False


@dataclass
class RequestOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    # bookkeeping fields
    input: RequestInput | None = None
    usage: dict[str, Any] | None = None

    completion_timestamp: float = 0.0  # Timestamp when the completion was generated

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

async def cornserve_invoke_eric(
    request_input: RequestInput,
    pbar: tqdm | None,
) -> RequestOutput:
    api_url = request_input.url
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        # only support images
        image_urls = get_image_data_uris(request_input.filenames)
        request_data = {
            "model_id": request_input.model,
            "data_urls": image_urls,
        }

        payload = {"request_data": request_data}

        output = RequestOutput()
        output.input = replace(request_input, multi_modal_data=[])
        output.prompt_len = request_input.prompt_len

        st = time.perf_counter()
        try:
            async with session.post( url=api_url, json=payload) as response:
                if response.status == 200:
                    # iterate over lines
                    timestamp = time.perf_counter()
                    output.latency = timestamp - st
                    output.completion_timestamp = timestamp
                    output.success = True
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
            "encoder_fission": request_input.encoder_fission,
            "ignore_eos": request_input.ignore_eos,
        }

        payload = {"request_data": request_data}

        output = RequestOutput()
        output.input = replace(request_input, multi_modal_data=[])
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
                        if not line:
                            # empty lines for keep alive
                            continue
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
                    output.completion_timestamp = most_recent_timestamp
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
            if np.random.rand() < config.encoder_fission_probability:
                # encoder fission
                request.encoder_fission = True
        # print(f"Total fissioned requests: {sum(request.encoder_fission for request in sampled_requests)}")

    request_inputs = []
    app_id = config.app_id
    for request in sampled_requests:
        mm_data_list = []
        if config.use_synthesized_data:
            for image_uri in request.image_urls:
                mm_data_list.append({"type": "image_url", "image_url": {"url": image_uri}})
        else:
            mm_data_list = [request.multi_modal_data]
        request_input = RequestInput(
            url=f"http://127.0.0.1:30080/app/invoke/{app_id}",
            model=config.model_id,
            prompt=request.prompt,
            prompt_len=request.prompt_len,
            output_len=config.output_len,
            multi_modal_data=mm_data_list,
            filenames=request.filenames,
            encoder_fission=request.encoder_fission,
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
            if isinstance(config.backend_config, EricConfig):
                return await cornserve_invoke_eric(request_input, pbar)
            return await cornserve_invoke(request_input, pbar)

    # first test a request
    print("Testing a single request to verify the setup...")
    test_output = await request_func(request_inputs[0], tqdm(total=1, desc="Test"))
    if not test_output.success:
        print("Test request failed. Please check the setup.")
        print(f"Error: {test_output.error}")
        raise RuntimeError("Test request failed.")

    # do warmup
    print("Starting warmup phase...")
    warmup_pbar = tqdm(total=config.num_warmups, desc="Warmup")
    coros = [request_func(request_input, warmup_pbar) for request_input in request_inputs[:config.num_warmups]]
    results = await asyncio.gather(*coros)
    if any(not output.success for output in results):
        print("Warmup requests failed. Please check the setup.")
        for output in results:
            if not output.success:
                print(f"Error: {output.error}")
        raise RuntimeError("Warmup requests failed.")

    print("=" * 50)

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
    
    # to rule out Timeout errors, we use the latest completion time
    benchmark_end_time = max([output.completion_timestamp for output in results if output.success])
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
        "{:<40} {:<10.4f}".format(
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

