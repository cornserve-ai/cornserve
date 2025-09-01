import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EricConfig, ExperimentConfig, VLLMConfig, EPDConfig, PDConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit


async def run(
    overwrite: bool = False,
) -> None:
    model_ids = []
    # model_ids.append("OpenGVLab/InternVL3-8B")
    model_ids.append("Qwen/Qwen2.5-VL-7B-Instruct")

    gpu_type="A100"
    MAX_BATCH_SIZE = 8
    REQUEST_RATE = 30

    app_ids = {}
    for model_id in model_ids:
        group = {}
        for i in range(1, MAX_BATCH_SIZE + 1):
        # for i in range(8,9):
            app_id = register_app(model_id=model_id, app_type="e", eric_max_batch_size=i)
            group[i] = app_id
            print(f"Registered {model_id} E with ID: {app_id} with batch size {i}")
        app_ids[model_id] = group

    configs = []
    image_width = 576
    image_height = 432
    image_count = 8
    num_prompts = 1000

    for model_id in model_ids:
        for bs, app_id in app_ids[model_id].items():
            exp_config = ExperimentConfig(
                backend_config=EricConfig(num_replicas=1, tp_size=1, max_batch_size=bs),
                app_id=app_id,
                model_id=model_id,
                request_rate=REQUEST_RATE,
                # Dedicated Eric profile
                input_len=0,
                output_len=0,
                image_count=image_count,
                num_prompts=num_prompts,
                image_width=image_width,
                image_height=image_height,
                image_choices=32,
                gpu_type=gpu_type,
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

    print(f"Sampling reqeuests ...")
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
    set_ulimit()
    await run(overwrite=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
