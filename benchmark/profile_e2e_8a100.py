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
    # model_ids = ["OpenGVLab/InternVL3-38B"]
    # model_ids = ["Qwen/Qwen2.5-VL-32B-Instruct"]
    model_ids = ["OpenGVLab/InternVL3-38B", "Qwen/Qwen2.5-VL-32B-Instruct"]

    app_types: list[Literal['ev', 'v', 'e', 'epd', 'pd', 'nccl-pd']] = ["ev", "v", "pd", "epd"]
    # app_types: list[Literal['ev', 'v', 'e', 'epd', 'pd', 'nccl-pd']] = ["epd"]
    # image_width, image_height, image_count, input_len, output_len, num_prompts, request_rate
    # the request rate is different per model!
    workloads = [
        # ----- Mono is better
        (1920, 1080, 1, 100, 100, 2000, 6),
        (1920, 1080, 1, 100, 300, 2000, 4.5),

        # ----- EV intern wins
        (1920, 1080, 1, 1000, 300, 2000, 4),
        # -----
        (1680, 1050, 1, 1000, 300, 2000, 5),
        # ----- For completeness
        (1680, 1050, 1, 100, 300, 2000, 5.5),
        # ----- ? Tie, 3.5 too much but 3 is not high enough
        (1680, 1050, 1, 1000, 500, 2000, 3.5),

        # ----- ?
        # scale up numbers
        (896, 896, 1, 100, 100, 2000, 15),
        (896, 896, 1, 1000, 300, 2000, 6),
        (896, 896, 2, 1000, 300, 2000, 4),

        # ----- small ones
        (512, 512, 1, 100, 100, 4000, 30),
        (512, 512, 1, 100, 300, 4000, 20),
        # -----
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
        num_erics = (8 // 2 - num_vllms) * 2
        cornserve_config = CornserveConfig(num_vllms=num_vllms, vllm_tp_size=2, num_erics=num_erics)
        backend_configs.append(cornserve_config)
    for num_prefills in [2]:
        num_decodes = 8//2 - num_prefills
        pd_config = PDConfig(num_prefills=num_prefills, num_decodes=num_decodes, prefill_tp_size=2, decode_tp_size=2)
        backend_configs.append(pd_config)
    for num_erics in [2]:
        # remaining = (8 - num_erics) // 2
        for num_prefills in [1]:
            num_decodes = 4 - num_erics//2 - num_prefills
            epd_config = EPDConfig(num_erics=num_erics, num_prefills=num_prefills, num_decodes=num_decodes, prefill_tp_size=2, decode_tp_size=2)
            backend_configs.append(epd_config)

    gpu_type = "A100"

    configs = []
    for model_id in model_ids:
        for backend in backend_configs:
            for workload in workloads:
                image_width, image_height, image_count, input_len, output_len, num_prompts, request_rate = workload
                if isinstance(backend, CornserveConfig):
                    app_type = "ev"
                elif isinstance(backend, VLLMConfig):
                    app_type = "v"
                elif isinstance(backend, PDConfig):
                    app_type = "pd"
                elif isinstance(backend, EPDConfig):
                    app_type = "epd"
                else:
                    raise NotImplementedError(f"Backend {backend} is not supported.")
                if app_type not in app_types:
                    continue
                app_id = app_ids[(model_id, app_type)]
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
                    use_synthesized_data=True
                )
                configs.append(exp_config)

    if not overwrite:
        configs = [cfg for cfg in configs if not cfg.exists()]

    # prioritize by request rate
    configs.sort(key=lambda config: (-config.request_rate,))

    print(f"Total configs: {len(configs)}")

    tokenizers = {}
    for model_id in model_ids:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            tokenizer_mode="auto",
            trust_remote_code=True,
        )
        tokenizers[model_id] = tokenizer

    sampled_workloads = {}
    def _try_sample_workload(cfg: ExperimentConfig) -> list[SampleRequest]:
        key = (cfg.model_id, cfg.num_prompts, cfg.output_len, cfg.input_len, cfg.seed)
        if key in sampled_workloads:
            return sampled_workloads[key]
        print(f"Sampling reqeuests ...")
        tokenizer = tokenizers[cfg.model_id]
        workload: list[SampleRequest] = VisionArenaDataset(
            dataset_path="lmarena-ai/VisionArena-Chat",
            dataset_subset=None,
            dataset_split="train",
            random_seed=cfg.seed,
        ).sample(
            num_requests=cfg.num_prompts,
            tokenizer=tokenizer,
            output_len=cfg.output_len,
            input_len=cfg.input_len,
        )
        sampled_workloads[key] = workload
        return workload


    for i, cfg in enumerate(configs):
        print(f"Current {i}/{len(configs)} config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        # we scale every time to clean up the task executors states just in case
        sampled_requests = _try_sample_workload(cfg)
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
