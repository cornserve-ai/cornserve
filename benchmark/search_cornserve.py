import argparse
import asyncio
import datetime
import json
import logging
import os
import requests

import numpy as np
from transformers import AutoTokenizer

from benchmark_backend import build_cornserve_vlm_input
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_benchmark import benchmark, post
from utils import get_benchmark_filenames, get_image_data_uris

# set to error
logging.basicConfig(level=logging.ERROR)

CONFIG = {
    "backend_url": "http://localhost:30080",
    "num_prompts": 2000,
    "output_len": 300,
    "enforced_prompt_len": 1500,
    "seed": 48105,
    "max_mm_count": 10, # random 10 images
    "burstiness": 1, # Poisson process
    "request_rate": 10, # fixed request rate
    "num_gpus": 12, # the config range to search from
    "starting_eric": 8,
}
OUTPUT_DIR = "results_search_cornserve"

async def main(args: argparse.Namespace) -> None:
    # first we check app id, error out of not present
    raw_response = requests.get(f"{CONFIG['backend_url']}/app/list")
    raw_response.raise_for_status()
    response: dict[str, str] = raw_response.json()
    if len(response.values()) != 1:
        raise RuntimeError("Expected exactly one app ID, found multiple or none.")
    app_id = list(response.keys())[0]
    app_status = list(response.values())[0]
    if app_status != "ready":
        raise RuntimeError(f"App ID {app_id} is not ready, current status: {app_status}")
    print(f"Using app ID: {app_id}")

    # now we get the task ids of eric and vLLM
    raw_response = requests.get(f"{CONFIG['backend_url']}/tasks/list")
    raw_response.raise_for_status()
    tasks_response = raw_response.json()
    if len(tasks_response) != 2:
        raise RuntimeError("Expected exactly two tasks, found multiple or none.")
    task_ids = [res[1] for res in tasks_response]
    for task_id in task_ids:
        if "encodertask" in task_id:
            encoder_task_id = task_id
        elif "llmtask" in task_id:
            llm_task_id = task_id
    if not encoder_task_id or not llm_task_id:
        raise RuntimeError("Could not find encoder or LLM task ID in the response.")
    print(f"Encoder Task ID: {encoder_task_id}")
    print(f"LLM Task ID: {llm_task_id}")

    # now we scale down the tasks to 0
    scale_endpoint = f"{CONFIG['backend_url']}/task/scale"
    while True:
        print(f"Scaling down encoder task {encoder_task_id}...")
        raw_response = requests.post(
            scale_endpoint,
            json={
                "task_id": encoder_task_id,
                "num_gpus": -1,
            },
        )
        if raw_response.status_code != 200:
            print(f"response: {raw_response.text}")
            break
    while True:
        print(f"Scaling down LLM task {llm_task_id}...")
        raw_response = requests.post(
            scale_endpoint,
            json={
                "task_id": llm_task_id,
                "num_gpus": -1,
            },
        )
        if raw_response.status_code != 200:
            print(f"response: {raw_response.text}")
            break

    def scale_task_with_num_replicas(task_id: str, num_replicas: int) -> None:
        print(f"Scaling task {task_id} with {num_replicas} replicas...")
        raw_response = requests.post(
            scale_endpoint,
            json={
                "task_id": task_id,
                "num_gpus": num_replicas,
            },
        )
        raw_response.raise_for_status()
        print(f"Scaled task {task_id} with {num_replicas} replicas.")

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
    for request in sampled_requests:
        chosen_image_filenames = list(np.random.choice(
            image_filenames,
            size=1,
            replace=False,
        ))
        request.filenames = chosen_image_filenames
        request.image_urls = get_image_data_uris(chosen_image_filenames)

    request_inputs = []
    for req in sampled_requests:
        request_input = build_cornserve_vlm_input(
            base_url=CONFIG["backend_url"],
            app_id=app_id,
            model_id=model_id,
            sampled_request=req,
            use_sampled_mm_data=False,
            video_urls=req.video_urls,
            audio_urls=req.audio_urls,
            image_urls=req.image_urls,
        )
        request_inputs.append(request_input)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datetime_str = datetime.datetime.now().astimezone().strftime("%m%d_%H%M")

    all_results = {}
    try:
        for i in range(CONFIG["starting_eric"], CONFIG["num_gpus"]):
            eric_replicas = i
            llm_replicas = CONFIG["num_gpus"] - i

            scale_task_with_num_replicas(encoder_task_id, eric_replicas)
            scale_task_with_num_replicas(llm_task_id, llm_replicas)

            if i == 1:
                print("Sending test request...")
                result = await post(request_input=request_inputs[0], pbar=None)
                if not result.success:
                    print("Test request failed with error:", result.error)
                    # reset
                    scale_task_with_num_replicas(encoder_task_id, -eric_replicas)
                    scale_task_with_num_replicas(llm_task_id, -llm_replicas)
                    exit(1)
                if args.test_only:
                    print("Sent", request_inputs[0])
                    print("Test request succeeded, result:", result)
                    scale_task_with_num_replicas(encoder_task_id, -eric_replicas)
                    scale_task_with_num_replicas(llm_task_id, -llm_replicas)
                    exit(0)

            # warm up 30 requests
            print("Warming up with 30 requests...")
            warmup_requests = request_inputs[:30]
            coros = [post(request_input=req, pbar=None) for req in warmup_requests]
            await asyncio.gather(*coros)
            print("Warm up completed.")
            
            benchmark_results = await benchmark(
                request_inputs=request_inputs,
                backend="cornserve",
                num_prompts=CONFIG["num_prompts"],
                request_rate=CONFIG["request_rate"],
                burstiness=CONFIG["burstiness"],
                max_concurrency=None,
                disable_tqdm=False,
            )
            all_results[f"{eric_replicas}eric_{llm_replicas}vllm"] = benchmark_results
            output_filename = (f"n{CONFIG['num_prompts']}_input{CONFIG['enforced_prompt_len']}_"
                               f"eric{eric_replicas}_vllm{llm_replicas}_"
                               f"output{CONFIG['output_len']}_burstiness{CONFIG['burstiness']}_"
                               f"seed{CONFIG['seed']}_rate{CONFIG['request_rate']}_{datetime_str}.json")
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            
            with open(output_filepath, 'w') as f:
                json.dump(benchmark_results , f, indent=2)

            ##### Comment out the next line to run all iterations
            exit(1)

            scale_task_with_num_replicas(encoder_task_id, -eric_replicas)
            scale_task_with_num_replicas(llm_task_id, -llm_replicas)
            await asyncio.sleep(5)
    except Exception as e:
        print(f"An error occurred during benchmarking: {e}")
        exit(1)

    # save all results in a single file
    all_results_filepath = os.path.join(
        OUTPUT_DIR,
        f"all_results_{datetime_str}_gpus{CONFIG['num_gpus']}"
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
