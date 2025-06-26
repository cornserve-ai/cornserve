import argparse
import asyncio
import contextlib
from datetime import datetime
import json
import os
import sys
import time
from typing import AsyncGenerator
import uuid

import aiohttp
from cornserve.services.gateway.models import AppInvocationRequest
import numpy as np
from transformers import AutoTokenizer

from benchmark_dataset import SampleRequest, VisionArenaDataset
from utils import get_benchmark_filenames
try:
    GATEWAY_URL = os.environ["CORNSERVE_GATEWAY_URL"]
except KeyError:
    print(
        "Environment variable CORNSERVE_GATEWAY_URL is not set. Defaulting to http://localhost:30080.\n",
    )
    GATEWAY_URL = "http://localhost:30080"

FILE_SERVER_URL = "http://ampere00.eecs.umich.edu:32000"
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
MAX_MM_COUNT = 40

video_filenames, audio_filenames, image_filenames = get_benchmark_filenames(MAX_MM_COUNT)
video_urls = [f"{FILE_SERVER_URL}/videos/{filename}" for filename in video_filenames]
audio_urls = [f"{FILE_SERVER_URL}/audios/{filename}" for filename in audio_filenames]
image_urls = [f"{FILE_SERVER_URL}/images/{filename}" for filename in image_filenames]

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

async def invoke(
    app_id: str,
    sampled_request: SampleRequest,
    use_sampled_mm_data: bool = True,
    video_urls: list[str] = [],
    audio_urls: list[str] = [],
    image_urls: list[str] = [],
):
    # data={"prompt": prompt, "multimodal_data": [], "return_audio": True}
    data={"prompt": sampled_request.prompt, "multimodal_data": []}
    if not use_sampled_mm_data:
        if len(video_urls):
            for u in video_urls:
                data["multimodal_data"].append(("video", u))
        if len(audio_urls):
            for u in audio_urls:
                data["multimodal_data"].append(("audio", u))
        if len(image_urls):
            for u in image_urls:
                data["multimodal_data"].append(("image", u))
    else:
        # sampled multimedia data only has one image in the VisionArena dataset
        assert sampled_request.multi_modal_data is not None, "SampleRequest missing multi_modal_data"
        data["multimodal_data"].append(("image", sampled_request.multi_modal_data["image_url"]["url"]))


    data["max_completion_tokens"] = sampled_request.expected_output_len
    request = AppInvocationRequest(request_data=data)
    api_url = f"{GATEWAY_URL}/app/invoke/{app_id}"
    result = {
        "success": False,
        "video_urls": video_urls,
        "audio_urls": audio_urls,
        "image_urls": image_urls,
    }
    start_time = time.perf_counter()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(api_url, json=request.model_dump()) as response:
                end_time = time.perf_counter()
                if response.status == 200:
                    result["success"] = True
                    result["latency"] = end_time - start_time
                    print(f"Response received in {result['latency']:.2f} seconds")
        except Exception:
            print(sys.exc_info()[0])
            result["latency"] = 0.0
    return result

async def post_to_eric(
    sampled_request: SampleRequest,
    use_sampled_mm_data: bool = True,
    video_urls: list[str] = [],
    audio_urls: list[str] = [],
    image_urls: list[str] = [],
) -> dict:
    base_url = "http://localhost:7999"
    api_url = f"{base_url}/embeddings"

    payload = {"data": []}
    if not use_sampled_mm_data:
        for url in video_urls:
            payload["data"].append({
                "id": uuid.uuid4().hex,
                "modality": "video",
                "url": url,
            })
        for url in audio_urls:
            payload["data"].append({
                "id": uuid.uuid4().hex,
                "modality": "audio",
                "url": url,
            })
        for url in image_urls:
            payload["data"].append({
                "id": uuid.uuid4().hex,
                "modality": "image",
                "url": url,
            })
    else:
        # sampled multimedia data only has one image in the VisionArena dataset
        assert sampled_request.multi_modal_data is not None, "SampleRequest missing multi_modal_data"
        url = sampled_request.multi_modal_data["image_url"]["url"]
        payload["data"].append({
            "id": uuid.uuid4().hex,
            "modality": "image",
            "url": url,
        })

    result = {
        "success": False,
        "video_urls": video_urls,
        "audio_urls": audio_urls,
        "image_urls": image_urls,
    }
    start_time = time.perf_counter()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(api_url, json=payload) as response:
                end_time = time.perf_counter()
                if response.status == 200:
                    result["success"] = True
                    result["latency"] = end_time - start_time
                    print(f"Response received in {result['latency']:.2f} seconds")
        except Exception:
            print(sys.exc_info()[0])
            result["latency"] = 0.0
    return result

async def post_to_vllm(
    model_id: str,
    sampled_request: SampleRequest,
    use_sampled_mm_data: bool = True,
    video_urls: list[str] = [],
    audio_urls: list[str] = [],
    image_urls: list[str] = [],
) -> dict:
    base_url = "http://localhost:8000"
    api_url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type" : "text", "text": sampled_request.prompt},
                ]
            },
        ],
        "max_completion_tokens": sampled_request.expected_output_len,
    }
    if not use_sampled_mm_data:
        for url in video_urls:
            payload["messages"][0]["content"].append({
                "type": "video_url",
                "video_url": url,
            })
        for url in audio_urls:
            payload["messages"][0]["content"].append({
                "type": "audio_url",
                "audio_url": url,
            })
        for url in image_urls:
            payload["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": url,
            })
    else:
        # sampled multimedia data only has one image in the VisionArena dataset
        assert sampled_request.multi_modal_data is not None, "SampleRequest missing multi_modal_data"
        payload["messages"][0]["content"].append(sampled_request.multi_modal_data)

    result = {
        "success": False,
        "video_urls": video_urls,
        "audio_urls": audio_urls,
        "image_urls": image_urls,
    }
    start_time = time.perf_counter()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(api_url, json=payload) as response:
                end_time = time.perf_counter()
                if response.status == 200:
                    result["success"] = True
                    result["latency"] = end_time - start_time
                    print(f"Response received in {result['latency']:.2f} seconds")
        except Exception:
            print(sys.exc_info()[0])
            result["latency"] = 0.0
    return result

async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
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

async def benchmark(input_requests: list[SampleRequest], args) -> None:
    # semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else contextlib.nullcontext()
    #
    # async def limited_request_func(request_func_input, pbar):
    #     async with semaphore:
    #         return await request_func(request_func_input=request_func_input, pbar=pbar)
    
    # Start time for overall benchmark
    benchmark_start_time = time.perf_counter()
    """Run benchmark and collect statistics."""
    tasks = []
    
    print(f"Starting benchmark with {len(input_requests)} requests...")
    
    # Generate requests and create tasks
    async for request in get_request(input_requests, args.request_rate, args.burstiness):
        # Get the URLs for this request
        video_urls = getattr(request, 'video_urls', [])
        audio_urls = getattr(request, 'audio_urls', [])
        image_urls = getattr(request, 'image_urls', [])
        
        # Create task for this request
        if args.backend.lower() == "cornserve":
            task = asyncio.create_task(invoke(
                app_id=args.app_id,
                sampled_request=request,
                video_urls=video_urls,
                audio_urls=audio_urls,
                image_urls=image_urls
            ))
        elif args.backend.lower() == "eric":
            # For Eric backend, we use the post_to_eric function
            task = asyncio.create_task(post_to_eric(
                sampled_request=request,
                video_urls=video_urls,
                audio_urls=audio_urls,
                image_urls=image_urls
            ))
        elif args.backend.lower() == "vllm":
            # For vLLM backend, we use the post_to_vllm function
            task = asyncio.create_task(post_to_vllm(
                model_id=args.model_id,
                sampled_request=request,
                video_urls=video_urls,
                audio_urls=audio_urls,
                image_urls=image_urls
            ))
        else:
            raise ValueError(f"Unsupported backend: {args.backend}")

        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    benchmark_end_time = time.perf_counter()
    total_time = benchmark_end_time - benchmark_start_time
    
    # Filter out exceptions and process results
    errors = [r for r in results if isinstance(r, Exception)]
    for error in errors:
        print(f"Error occurred: {error}")
    valid_results = [r for r in results if isinstance(r, dict)]
    successful_requests = [r for r in valid_results if r.get("success", False)]
    failed_requests = [r for r in valid_results if not r.get("success", False)]
    latencies = [r["latency"] for r in successful_requests if r.get("latency", 0) > 0]
    
    # Calculate statistics
    stats = {
        "total_requests": len(valid_results),
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "success_rate": len(successful_requests) / len(valid_results) if valid_results else 0,
        "total_time": total_time,
        "actual_request_rate": len(valid_results) / total_time if total_time > 0 else 0,
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
    
    output_filename = f"benchmark_results_{args.backend}_{args.num_prompts}_{int(time.time())}.json"

    output_filename = f"benchmark_results_{args.backend}_{args.num_prompts}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_data = {
        "statistics": stats,
        "detailed_results": valid_results,
        "benchmark_config": {
            "seed": args.seed,
            "request_rate": args.request_rate,
            "burstiness": args.burstiness,
            "num_prompts": args.num_prompts,
            "video_prob": args.video_prob,
            "audio_prob": args.audio_prob,
            "image_prob": args.image_prob,
            "max_videos_per_request": args.max_videos_per_request,
            "max_audios_per_request": args.max_audios_per_request,
            "max_images_per_request": args.max_images_per_request,
            "mm_distribution": args.mm_distribution,
        }
    }
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nBenchmark completed!")
    print(f"Results saved to: {output_filename}")
    print(f"\n=== BENCHMARK STATISTICS ===")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Successful requests: {stats['successful_requests']}")
    print(f"Failed requests: {stats['failed_requests']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Actual request rate: {stats['actual_request_rate']:.2f} req/s")
    
    if latencies:
        print(f"\n=== LATENCY STATISTICS ===")
        print(f"Mean latency: {stats['latency_mean']:.3f}s")
        print(f"Median latency: {stats['latency_median']:.3f}s")
        print(f"95th percentile: {stats['latency_p95']:.3f}s")
        print(f"99th percentile: {stats['latency_p99']:.3f}s")
        print(f"Min latency: {stats['latency_min']:.3f}s")
        print(f"Max latency: {stats['latency_max']:.3f}s")


async def main(args: argparse.Namespace) -> None:
    model_id = args.model_id
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, tokenizer_mode = "auto", trust_remote_code=True)

    # Currently default to the VisionArena dataset
    input_requests: list[SampleRequest]= VisionArenaDataset(
        dataset_path="lmarena-ai/VisionArena-Chat",
        dataset_subset=None,
        dataset_split="train",
        random_seed=args.seed,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
        )

    backend = args.backend.lower()
    if backend == "cornserve":
        response = await invoke(args.app_id, input_requests[0])
        print(f"Test response from Cornserve: {response}")
    elif backend == "vllm":
        response = await post_to_vllm(model_id=model_id, sampled_request=input_requests[0])
        print(f"Test response from vLLM: {response}")
    elif backend == "eric":
        response = await post_to_eric(sampled_request=input_requests[0])
        print(f"Test response from Eric: {response}")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    if args.test_only:
        print("Test completed successfully. Exiting without running full benchmark.")
        return

    # Add random multimedia URLs to each request
    for request in input_requests:
        # Decide whether to include each media type
        include_videos = np.random.random() < args.video_prob
        include_audios = np.random.random() < args.audio_prob  
        include_images = np.random.random() < args.image_prob
    
        # Sample number of each media type
        if include_videos and args.max_videos_per_request > 0 and len(video_urls) > 0:
            num_videos = sample_mm_count(args.max_videos_per_request, args.mm_distribution)
            if num_videos > 0:
                request.video_urls = np.random.choice(video_urls, 
                                                    size=min(num_videos, len(video_urls)), 
                                                    replace=False).tolist()  # type: ignore
        
        if include_audios and args.max_audios_per_request > 0 and len(audio_urls) > 0:
            num_audios = sample_mm_count(args.max_audios_per_request, args.mm_distribution)
            if num_audios > 0:
                request.audio_urls = np.random.choice(audio_urls,
                                                    size=min(num_audios, len(audio_urls)),
                                                    replace=False).tolist()  # type: ignore
            
        if include_images and args.max_images_per_request > 0 and len(image_urls) > 0:
            num_images = sample_mm_count(args.max_images_per_request, args.mm_distribution)
            if num_images > 0:
                request.image_urls = np.random.choice(image_urls,
                                                    size=min(num_images, len(image_urls)),
                                                    replace=False).tolist()  # type: ignore

    print(f"Sample requests prepared: {len(input_requests)}")
    await benchmark(input_requests, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cornserve e2e benchmark")
    parser.add_argument("--backend", type=str, default="cornserve", choices=["cornserve", "eric", "vllm"], help="Backend to use for the benchmark.")
    parser.add_argument("--app-id", type=str, help="App ID to invoke")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model ID to use for the benchmark.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Request rate in requests per second.")
    parser.add_argument("--burstiness", type=float, default=1.0, 
                        help="Burstiness factor for request generation.")
    parser.add_argument("--num-prompts", type=int, default=1000, help="Number of prompts to process.")
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
