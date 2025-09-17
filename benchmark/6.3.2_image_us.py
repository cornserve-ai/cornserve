import asyncio

from transformers import AutoTokenizer

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, SyntheticDiffusionDataset, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import ExperimentConfig, CornserveQwenImageConfig, QwenImageConfig, VLLMConfig

from cornserve.utils import set_ulimit

async def run(
    overwrite: bool = False,
) -> None:
    model_id = "Qwen/Qwen-Image"

    qwen_image_app = register_app(
        model_id=model_id,
        app_type="qwen-image",
    )
    print(f"Registered app {qwen_image_app}")

    backend = CornserveQwenImageConfig(
        num_llms=1,
        num_geris=7,
    )

    gpu_type = "A100"

    input_len = 1000
    # Diffusion DB has very short text input length due to context length limit
    # input_len = 75
    num_requests = 1000
    request_rate = 5
    output_image_width = 512
    output_image_height = 512
    num_inference_steps = 20

    workload_config = QwenImageConfig(
        output_image_width=output_image_width,
        output_image_height=output_image_height,
        num_inference_steps=num_inference_steps,
    )

    image_sampled_requests = SyntheticDiffusionDataset(48105).sample(
        num_requests=num_requests,
        input_len=input_len,
        image_width=output_image_width,
        image_height=output_image_height,
        num_inference_steps=num_inference_steps,
    )

    configs = []

    exp_config = ExperimentConfig(
        backend_config=backend,
        app_id=qwen_image_app,
        model_id=model_id,
        gpu_type=gpu_type,
        dataset="synthetic",
        workload_config=workload_config,
        request_rate=request_rate,
        use_synthesized_data=False,
        num_warmups=20,
    )
    configs.append((exp_config, image_sampled_requests))

    # prioritize by request rate
    configs.sort(key=lambda config: (-config[0].request_rate,))
    configs = configs if overwrite else [cfg for cfg in configs if not cfg[0].exists()]

    print(f"Total configs: {len(configs)}")

    for cfg, sampled_requests in configs:
        print(f"Current config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        # we scale every time to clean up the task executors states just in case
        await scale(cfg)
        request_inputs = transform_sampled_requests(config=cfg, sampled_requests=sampled_requests)
        output_data = await benchmark(request_inputs=request_inputs, config=cfg)
        completed = output_data["metrics"]["completed"]
        if completed <= len(sampled_requests) * 0.95:
            raise RuntimeError("Insufficient completed requests")
        print("Benchmark completed for current batch.")
        print("=" * 50)

async def main():
    set_ulimit()
    await run(overwrite=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")



