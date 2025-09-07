"""Execute throughput benchmark for InternVL3-38B model."""

import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import EPDConfig, ExperimentConfig, VLLMConfig, PDConfig, CornserveConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit


async def run(
    overwrite: bool = False,
) -> None:
    """Run throughput benchmark for InternVL3-38B model."""
    model_id: str = "OpenGVLab/InternVL3-38B"
    pd = register_app(model_id=model_id, app_type="pd")
    print(f"Registered {model_id} pd with ID: {pd}")
    vllm = register_app(model_id=model_id, app_type="v")
    print(f"Registered {model_id} V with ID: {vllm}")
    ev = register_app(model_id=model_id, app_type="ev")
    print(f"Registered {model_id} EV with ID: {ev}")
    epd = register_app(model_id=model_id, app_type="epd")
    print(f"Registered {model_id} epd with ID: {epd}")

    vllm_config = VLLMConfig(num_replicas=4, tp_size=2)
    pd_config = PDConfig(num_prefills=2, prefill_tp_size=2, num_decodes=2, decode_tp_size=2)
    ev_config = CornserveConfig(num_vllms=3, vllm_tp_size=2, num_erics=2)
    epd_d_config = EPDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=2, decode_tp_size=2, num_erics=2)

    configs = []
    gpu_type = "A100"
    image_width = 1920
    image_height = 1080
    image_count = 1
    input_len = 100
    output_len = 300
    num_prompts = 1000

    # request_rates = [1, 1.5, 2]
    request_rates = [1.5]

    # mono
    for r in request_rates:
        exp_config = ExperimentConfig(
            backend_config=vllm_config,
            app_id=vllm,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(exp_config)
    # pd
    for r in request_rates:
        exp_config = ExperimentConfig(
            backend_config=pd_config,
            app_id=pd,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(exp_config)
    # el
    for r in request_rates:
        exp_config = ExperimentConfig(
            backend_config=ev_config,
            app_id=ev,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        # configs.append(exp_config)
    # epd
    for r in request_rates:
        exp_config = ExperimentConfig(
            backend_config=epd_d_config,
            app_id=epd,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        # configs.append(exp_config)

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
