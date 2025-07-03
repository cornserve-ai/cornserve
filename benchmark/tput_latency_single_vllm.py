import argparse
import asyncio
import datetime
import json
import json
import logging
import os

import numpy as np
from transformers import AutoTokenizer

from benchmark_backend import build_vllm_input
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_benchmark import benchmark, post
from utils import get_benchmark_filenames, get_image_data_uris

# set to error
logging.basicConfig(level=logging.ERROR)

CONFIG = {
    "backend_url": "http://localhost:8000",
    "num_prompts": 2000,
    "output_len": 300,
    "enforced_prompt_len": 1500,
    "seed": 48105,
    "max_mm_count": 10, # random 40 images
    "burstiness": 1, # Poisson process
    "max_request_rate": 1, # max requests per second
}
OUPTUT_DIR = "results_tput_latency_single_vllm"

async def main(args: argparse.Namespace) -> None:
    model_id = args.model_id
    np.random.seed(args.seed)

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

    print("Sending test request...")
    result = await post(request_input=request_inputs[0], pbar=None)
    if not result.success:
        print("Test request failed with error:", result.error)
        exit(1)
    if args.test_only:
        print("Sent", request_inputs[0])
        print("Test request succeeded, result:", result)
        exit(0)

    request_rates = list(range(1, CONFIG["max_request_rate"] + 1))

    os.makedirs(OUPTUT_DIR, exist_ok=True)
    datetime_str = datetime.datetime.now().strftime("%m%d_%H%M")

    all_results = {}
    for request_rate in request_rates:
        benchmark_results = await benchmark(
            request_inputs=request_inputs,
            backend="vllm",
            num_prompts=CONFIG["num_prompts"],
            request_rate=request_rate,
            burstiness=CONFIG["burstiness"],
            max_concurrency=None,
            disable_tqdm=False,
        )
        all_results[request_rate] = benchmark_results
        output_filename = (f"n{CONFIG['num_prompts']}_input{CONFIG['enforced_prompt_len']}_"
                           f"output{CONFIG['output_len']}_burstiness{CONFIG['burstiness']}_"
                           f"seed{args.seed}_rate{request_rate}_{datetime_str}.json")
        output_filepath = os.path.join(OUPTUT_DIR, output_filename)
        
        with open(output_filepath, 'w') as f:
            json.dump(benchmark_results , f, indent=2)

    # save all results in a single file
    all_results_filepath = os.path.join(
        OUPTUT_DIR,
        f"all_results_{datetime_str}_request_rates_{min(request_rates)}_to_{max(request_rates)}_"
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
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="Random seed for sampling requests")
    
    args = parser.parse_args()
    asyncio.run(main(args))
