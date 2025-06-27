import argparse
import asyncio
import contextlib
import json
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, AsyncGenerator, Callable

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from benchmark_backend import TRANSFORM_FUNCS, RequestInput, RequestOutput
from benchmark_dataset import FILE_SERVER_URL, SampleRequest, VisionArenaDataset
from utils import get_benchmark_filenames

GATEWAY_URL = "http://localhost:30080"
BACKEND_URLS = {
    "cornserve": GATEWAY_URL,
    "vllm": "http://localhost:8000",
    "eric": "http://localhost:7999",
}
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
MAX_MM_COUNT = 40

def sample_mm_count(max_count: int, distribution: str = "poisson") -> int:
    """Sample number of multimedia items based on distribution."""
    if max_count <= 0:
        return 0
    
    if distribution == "uniform":
        return np.random.randint(0, max_count + 1)
    elif distribution == "poisson":
        # Use max_count/2 as lambda to keep most values within range
        lambda_val = max(0.5, max_count / 2.0)  # Ensure lambda is positive
        sampled = np.random.poisson(lambda_val)
        return min(sampled, max_count)
    elif distribution == "geometric":
        # Geometric with p = 0.3 to favor smaller numbers
        sampled = np.random.geometric(0.3) - 1  # -1 because geometric starts at 1
        return min(max(sampled, 0), max_count)  # Ensure non-negative
    else:
        return np.random.randint(0, max_count + 1)


async def post(
    request_input: RequestInput,
    pbar: tqdm | None,
) -> RequestOutput:
    """Send a POST request to the specified URL with the given payload."""
    result = RequestOutput()
    start_time = time.perf_counter()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(request_input.url, json=request_input.payload) as response:
                if response.status == 200:
                    content = await response.json()
                    end_time = time.perf_counter()
                    result.success = True
                    result.latency = end_time - start_time
                    result.output = content
                else:
                    result.error = f"Request failed with status {response.status}"
        except Exception as e:
            result.error = f"Request failed with exception: {str(e)}"
    if pbar:
        pbar.update(1)
    return result

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


async def benchmark(
    request_inputs: list[RequestInput],
    backend: str,
    num_prompts: int,
    request_rate: float,
    burstiness: float,
    max_concurrency: int | None,
    disable_tqdm: bool,
) -> dict[str, Any]:
    pbar = None if disable_tqdm else tqdm(total=len(request_inputs))
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()
    async def request_func(request_input: RequestInput, pbar: tqdm | None = None) -> RequestOutput:
        async with semaphore:
            return await post(request_input=request_input, pbar=pbar)
    
    """Run benchmark and collect statistics."""
    
    print(f"Starting benchmark with {len(request_inputs)} requests...")
    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")
    
    # Start time for overall benchmark
    benchmark_start_time = time.perf_counter()
    tasks = []
    # Generate requests and create tasks
    async for request in get_request(request_inputs, request_rate, burstiness):
        task = asyncio.create_task(request_func(request_input=request, pbar=pbar))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    benchmark_end_time = time.perf_counter()
    total_time = benchmark_end_time - benchmark_start_time
    
    successful_requests = [r for r in results if r.success]
    failed_requests = [r for r in results if not r.success]
    latencies = [r.latency for r in successful_requests if r.latency > 0]
    
    # Calculate statistics
    stats = {
        "total_requests": len(results),
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "success_rate": len(successful_requests) / len(results) if results else 0,
        "total_time": total_time,
        "actual_request_rate": len(results) / total_time if total_time > 0 else 0,
    }
    
    if latencies:
        stats.update({
            "latency_mean": np.mean(latencies),
            "latency_median": np.median(latencies),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "latency_min": np.min(latencies),
            "latency_max": np.max(latencies),
        })
    else:
        stats.update({
            "latency_mean": 0,
            "latency_median": 0,
            "latency_p95": 0,
            "latency_p99": 0,
            "latency_min": 0,
            "latency_max": 0,
        })

    output_data = {
        "statistics": stats,
        "detailed_results": [asdict(result) for result in results],
        "benchmark_config": {
            "request_rate": request_rate,
            "burstiness": burstiness,
            "num_prompts": num_prompts,
        }
    }

    return output_data
    


async def main(args: argparse.Namespace) -> None:
    model_id = args.model_id
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, tokenizer_mode = "auto", trust_remote_code=True)

    # Currently default to the VisionArena dataset
    sampled_requests: list[SampleRequest]= VisionArenaDataset(
        dataset_path="lmarena-ai/VisionArena-Chat",
        dataset_subset=None,
        dataset_split="train",
        random_seed=args.seed,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
        )

    # Add random multimedia URLs to each request
    if args.synthesize_mm_data:
        video_filenames, audio_filenames, image_filenames = get_benchmark_filenames(MAX_MM_COUNT)
        video_urls = [f"{FILE_SERVER_URL}/videos/{filename}" for filename in video_filenames]
        audio_urls = [f"{FILE_SERVER_URL}/audios/{filename}" for filename in audio_filenames]
        image_urls = [f"{FILE_SERVER_URL}/images/{filename}" for filename in image_filenames]

        for request in sampled_requests:
            # Decide whether to include each media type
            include_videos = np.random.random() < args.video_prob
            include_audios = np.random.random() < args.audio_prob  
            include_images = np.random.random() < args.image_prob
        
            # Sample number of each media type
            if include_videos and args.max_videos_per_request > 0 and len(video_urls) > 0:
                num_videos = sample_mm_count(args.max_videos_per_request, args.mm_distribution)
                if num_videos > 0:
                    request.video_urls = np.random.choice(
                        video_urls,
                        size=min(num_videos, len(video_urls)),
                        replace=False,
                    ).tolist()  # type: ignore
            
            if include_audios and args.max_audios_per_request > 0 and len(audio_urls) > 0:
                num_audios = sample_mm_count(args.max_audios_per_request, args.mm_distribution)
                if num_audios > 0:
                    request.audio_urls = np.random.choice(
                        audio_urls,
                        size=min(num_audios, len(audio_urls)),
                        replace=False,
                    ).tolist()  # type: ignore
                
            if include_images and args.max_images_per_request > 0 and len(image_urls) > 0:
                num_images = sample_mm_count(args.max_images_per_request, args.mm_distribution)
                if num_images > 0:
                    request.image_urls = np.random.choice(
                        image_urls,
                        size=min(num_images, len(image_urls)),
                        replace=False,
                    ).tolist()  # type: ignore

    backend = args.backend.lower()
    if backend not in TRANSFORM_FUNCS:
        raise ValueError(f"Unsupported backend: {backend}")
    transform_func: Callable = TRANSFORM_FUNCS[backend]

    request_inputs = []
    for req in sampled_requests:
        request_input = transform_func(
            base_url=BACKEND_URLS[backend] if args.backend_url is None else args.backend_url,
            app_id=args.app_id,
            model_id=model_id,
            sampled_request=req,
            use_sampled_mm_data=not args.synthesize_mm_data,
            video_urls=req.video_urls,
            audio_urls=req.audio_urls,
            image_urls=req.image_urls,
        )
        request_inputs.append(request_input)

    print("Sending test request...")
    result = await post(request_input=request_inputs[0], pbar=None)
    if not result.success:
        print("Test request failed with error:", result.error)
        exit(1)
    if args.test_only:
        exit(0)

    benchmark_results = await benchmark(
        request_inputs=request_inputs,
        backend=args.backend,
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        max_concurrency=args.max_concurrency,
        disable_tqdm=args.disable_tqdm,
    )
    benchmark_results["args"] = vars(args)

    output_filename = (f"benchmark_{args.backend}_{args.num_prompts}_"
                       f"seed{args.seed}_request_rate{args.request_rate}_"
                       f"burstiness{args.burstiness}_synthesize{args.synthesize_mm_data}_"
                          f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_filename, 'w') as f:
        json.dump(benchmark_results , f, indent=2)
    
    print("\nBenchmark completed!")
    print(f"Results saved to: {output_filename}")
    print("\n=== BENCHMARK STATISTICS ===")
    print(f"Total requests: {benchmark_results['statistics']['total_requests']}")
    print(f"Successful requests: {benchmark_results['statistics']['successful_requests']}")
    print(f"Failed requests: {benchmark_results['statistics']['failed_requests']}")
    print(f"Success rate: {benchmark_results['statistics']['success_rate']:.2%}")
    print(f"Total time: {benchmark_results['statistics']['total_time']:.2f}s")
    print(f"Actual request rate: {benchmark_results['statistics']['actual_request_rate']:.2f} req/s")
    
    if benchmark_results['statistics']['successful_requests'] > 0:
        print("\n=== LATENCY STATISTICS ===")
        print(f"Mean latency: {benchmark_results['statistics']['latency_mean']:.3f}s")
        print(f"Median latency: {benchmark_results['statistics']['latency_median']:.3f}s")
        print(f"95th percentile: {benchmark_results['statistics']['latency_p95']:.3f}s")
        print(f"99th percentile: {benchmark_results['statistics']['latency_p99']:.3f}s")
        print(f"Min latency: {benchmark_results['statistics']['latency_min']:.3f}s")
        print(f"Max latency: {benchmark_results['statistics']['latency_max']:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cornserve e2e benchmark")
    parser.add_argument("--backend", type=str, default="cornserve", choices=["cornserve", "eric", "vllm"], help="Backend to use for the benchmark.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--backend-url", type=str, help="URL of the backend service.")
    parser.add_argument("--app-id", type=str, help="App ID to invoke")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model ID to use for the benchmark.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Request rate in requests per second.")
    parser.add_argument("--burstiness", type=float, default=1.0, 
                        help="Burstiness factor for request generation.")
    parser.add_argument("--num-prompts", type=int, default=100, help="Number of prompts to process.")
    parser.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    parser.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    parser.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )
    parser.add_argument("--video-prob", type=float, default=0, 
                        help="Probability of including videos in a request.")
    parser.add_argument("--audio-prob", type=float, default=0,
                        help="Probability of including audios in a request.")
    parser.add_argument("--image-prob", type=float, default=1,
                        help="Probability of including images in a request.")
    parser.add_argument("--max-videos-per-request", type=int, default=0,
                        help="Maximum number of videos per request.")
    parser.add_argument("--max-audios-per-request", type=int, default=0,
                        help="Maximum number of audios per request.")
    parser.add_argument("--max-images-per-request", type=int, default=3,
                        help="Maximum number of images per request.")
    parser.add_argument("--mm-distribution", type=str, default="poisson",
                        choices=["uniform", "poisson", "geometric"],
                        help="Distribution for number of multimedia items per request.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument("--test-only", action="store_true",
                        help="If set, only test the connection to the backend without running the full benchmark.")
    parser.add_argument("--synthesize-mm-data", action="store_true",
                        help="If set, synthesize multimedia data for each request instead of using the image in VisionArena dataset.")
    args = parser.parse_args()

    if args.backend == "cornserve" and not args.app_id:
        parser.error("--app-id is required when --backend is 'cornserve'")

    asyncio.run(main(args))
