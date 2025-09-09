"""Execute throughput benchmark for InternVL3-38B model."""

import asyncio

from typing import Literal

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EPDConfig, ExperimentConfig, PDConfig, VLLMConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit


async def run(
    overwrite: bool = False,
) -> None:
    """Sweep"""
    model_ids = ["OpenGVLab/InternVL3-38B", "Qwen/Qwen2.5-VL-32B-Instruct"]
    app_types: list[Literal['ev', 'v', 'e', 'epd', 'pd', 'nccl-pd']] = ["ev", "v"]
    # image_width, image_height, image_count, input_len, output_len, num_prompts, request_rate
    workloads = [
        (1920, 1080, 1, 100, 100, 2000, 10),
        (1920, 1080, 1, 100, 300, 2000, 10),
        (1920, 1080, 1, 1000, 100, 2000, 10),
        (1920, 1080, 1, 1000, 300, 2000, 10),
        (1680, 1050, 1, 100, 100, 2000, 10),
        (1680, 1050, 1, 100, 300, 2000, 10),
        (1680, 1050, 1, 1000, 100, 2000, 10),
        (1680, 1050, 1, 1000, 300, 2000, 10),
        (896, 896, 1, 100, 100, 2000, 10),
        (896, 896, 1, 100, 300, 2000, 10),
        (896, 896, 1, 1000, 100, 2000, 10),
        (896, 896, 1, 1000, 300, 2000, 10),
        (512, 512, 1, 100, 100, 2000, 10),
        (512, 512, 1, 100, 300, 2000, 10),
        (512, 512, 1, 1000, 100, 2000, 10),
        (512, 512, 1, 1000, 300, 2000, 10),
    ]
    # image tokens , Qwen2.5-VL-32B-Instruct, InternVL3-38B
    # (2400, 1800) ->  5504,                , 3328
    # (2800, 1200) ->  4300,                , 2816
    # (1920, 1080) ->  2691,                , 2304
    # (1680, 1050) ->  2280,                , 1792
    # (896, 896)   ->  1024,                , 1280
    # (512, 512)   ->  324,                 , 256

    # (2560, 1440) ->  4641,                , 2304
    # ...

    app_ids = {}
    for model_id in model_ids:
        for app_type in app_types:
            app_id = register_app(model_id=model_id, app_type=app_type)
            app_ids[(model_id, app_type)] = app_id
            print(f"Registered {model_id} {app_type} with ID: {app_id}")

    backend_configs = []
    vllm_config = VLLMConfig(num_replicas=4, tp_size=2)
    backend_configs.append(vllm_config)
    for num_vllms in [3]:
        num_erics = 8 // 2 - num_vllms
        cornserve_config = CornserveConfig(num_vllms=num_vllms, vllm_tp_size=2, num_erics=num_erics)
        backend_configs.append(cornserve_config)

    gpu_type = "A100"

    configs = []
    for model_id in model_ids:
        for backend in backend_configs:
            for workload in workloads:
                image_width, image_height, image_count, input_len, output_len, num_prompts, request_rate = workload
                app_id = app_ids[(model_id, "ev" if isinstance(backend, CornserveConfig) else "v")]
                exp_config = ExperimentConfig(
                    backend_config=backend,
                    app_id=app_id,
                    model_id=model_id,
                    request_rate=request_rate,
                    input_len=input_len,
                    output_len=output_len,
                    image_count=image_count,
                    num_prompts=num_prompts,
                    image_width=image_width,
                    image_height=image_height,
                    gpu_type = gpu_type,
                )
                configs.append(exp_config)

    if not overwrite:
        configs = [cfg for cfg in configs if not cfg.exists()]

    # prioritize by request rate
    configs.sort(key=lambda config: (-config.request_rate,))

    print(f"Total configs: {len(configs)}")

    shared_config = next(iter(configs))
    tokenizer = AutoTokenizer.from_pretrained(
        shared_config.model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )

    print("Sampling reqeuests ...")
    sampled_requests: list[SampleRequest] = VisionArenaDataset(
        dataset_path="lmarena-ai/VisionArena-Chat",
        dataset_subset=None,
        dataset_split="train",
        random_seed=shared_config.seed,
    ).sample(
        num_requests=shared_config.num_prompts,
        tokenizer=tokenizer,
        output_len=shared_config.output_len,
        input_len=shared_config.input_len,
    )

    for cfg in configs:
        print(f"Current config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        # we scale every time to clean up the task executors states just in case
        await scale(cfg)
        request_inputs = transform_sampled_requests(config=cfg, sampled_requests=sampled_requests)
        await benchmark(request_inputs=request_inputs, config=cfg)
        print("Benchmark completed for current batch.")
        print("=" * 50)


async def main():
    """Main function."""
    set_ulimit()
    await run(overwrite=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
