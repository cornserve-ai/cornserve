import argparse
import httpx
import asyncio
import datetime
import json
import logging
import os
import yaml
from multiprocessing import Process

import numpy as np
from transformers import AutoTokenizer

from benchmark_backend import build_vllm_input
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_benchmark import benchmark, post
from utils import get_benchmark_filenames, get_image_data_uris

# set to error
logging.basicConfig(level=logging.ERROR)

proxy_config = yaml.safe_load(open("proxy_config.yaml"))
PROXY_SERVERS = proxy_config.get("server_addrs")
if not PROXY_SERVERS:
    raise ValueError("No proxy servers found in proxy_config.yaml")

CONFIG = {
    "backend_url": "http://localhost:8000",
    "num_prompts": 2000,
    "output_len": 300,
    "enforced_prompt_len": 1500,
    "seed": 48105,
    "max_mm_count": 10, # random 10 images
    "burstiness": 1, # Poisson process
    "request_rate": 10, # fixed request rate
}
OUTPUT_DIR = "results_scaling_vllm"

def run_proxy(addrs: list[str], port: int = 8000) -> None:
    """Sync entry-point for multiprocessing; spins up the async proxy."""
    import asyncio
    from proxy_server_vllm import start_proxy_server
    asyncio.run(start_proxy_server(addrs, port))

async def main(args: argparse.Namespace) -> None:
    model_id = args.model_id
    np.random.seed(CONFIG["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, tokenizer_mode = "auto", trust_remote_code=True)
    sampled_requests: list[SampleRequest]= VisionArenaDataset(
        dataset_path="lmarena-ai/VisionArena-Chat",
        dataset_subset=None,
        dataset_split="train",
        random_seed=CONFIG["seed"],
        ).sample(
            num_requests=CONFIG["num_prompts"],
            tokenizer=tokenizer,
            output_len=CONFIG["output_len"],
            enforced_prompt_len=CONFIG["enforced_prompt_len"],
        )
    print("Synthesizing multimodal data...")
    _, _, image_filenames = get_benchmark_filenames(CONFIG["max_mm_count"])
    image_urls = get_image_data_uris(image_filenames)
    for request in sampled_requests:
        request.image_urls = np.random.choice(
            image_urls,
            size=1,
            replace=False,
        ).tolist()  # type: ignore

    request_inputs = []
    for req in sampled_requests:
        request_input = build_vllm_input(
            base_url=CONFIG["backend_url"],
            app_id="",
            model_id=model_id,
            sampled_request=req,
            use_sampled_mm_data=False,
            video_urls=req.video_urls,
            audio_urls=req.audio_urls,
            image_urls=req.image_urls,
        )
        request_inputs.append(request_input)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datetime_str = datetime.datetime.now().strftime("%m%d_%H%M")

    all_results = {}
    try:
        for i in range(1, len(PROXY_SERVERS) + 1):
            server_addrs = PROXY_SERVERS[:i]
            # start proxy server in a separate process

            server_process = Process(
                target=run_proxy,
                args=(server_addrs,)
            )
            server_process.start()

            # wait for server to healthy
            start = asyncio.get_event_loop().time()
            while True:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        response = await client.get(f"{CONFIG['backend_url']}/health")
                        if response.status_code == 200:
                            print(f"Proxy server {CONFIG['backend_url']} is healthy")
                            break
                except Exception as e:
                    print(f"Waiting for proxy server {CONFIG['backend_url']} to be healthy: {e}")
                await asyncio.sleep(1)
                if (asyncio.get_event_loop().time() - start) > 60:
                    print("Timed out waiting for proxy server to be healthy")
                    if server_process.is_alive():
                        server_process.terminate()
                    server_process.join()
                    raise RuntimeError("Proxy server did not become healthy in time")

            if i == 1:
                print("Sending test request...")
                result = await post(request_input=request_inputs[0], pbar=None)
                if not result.success:
                    print("Test request failed with error:", result.error)
                    # kill the server process
                    if server_process.is_alive():
                        server_process.terminate()
                    server_process.join()
                    exit(1)
                if args.test_only:
                    print("Sent", request_inputs[0])
                    print("Test request succeeded, result:", result)
                    # kill the server process
                    if server_process.is_alive():
                        server_process.terminate()
                    server_process.join()
                    exit(0)
            
            benchmark_results = await benchmark(
                request_inputs=request_inputs,
                backend="vllm",
                num_prompts=CONFIG["num_prompts"],
                request_rate=CONFIG["request_rate"],
                burstiness=CONFIG["burstiness"],
                max_concurrency=None,
                disable_tqdm=False,
            )
            all_results[len(server_addrs)] = benchmark_results
            output_filename = (f"n{CONFIG['num_prompts']}_input{CONFIG['enforced_prompt_len']}_"
                               f"instances{len(server_addrs)}_"
                               f"output{CONFIG['output_len']}_burstiness{CONFIG['burstiness']}_"
                               f"seed{CONFIG['seed']}_rate{CONFIG['request_rate']}_{datetime_str}.json")
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            
            with open(output_filepath, 'w') as f:
                json.dump(benchmark_results , f, indent=2)


            # kill the server process
            if server_process.is_alive():
                server_process.terminate()
            server_process.join()
            await asyncio.sleep(5)
    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")
        exit(1)

    # save all results in a single file
    all_results_filepath = os.path.join(
        OUTPUT_DIR,
        f"all_results_{datetime_str}_replicas{len(PROXY_SERVERS)}_"
        f"{CONFIG['num_prompts']}_"
        f"input{CONFIG['enforced_prompt_len']}_output{CONFIG['output_len']}.json"
    )
    with open(all_results_filepath, 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark vLLM with VisionArena dataset")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model ID to benchmark",
    )
    parser.add_argument("--test-only", action="store_true", help="Run a test request only")
    
    args = parser.parse_args()
    asyncio.run(main(args))
