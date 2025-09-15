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
    vllm_app = register_app(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        app_type="v",
    )
    print(f"Registered app {vllm_app}")

    backend = CornserveQwenImageConfig(
        num_llms=7,
        num_geris=1
    )

    llm_backend = VLLMConfig(
        num_replicas=1,
    )

    gpu_type = "A100"

    input_len = 1000
    # Diffusion DB has very short text input length due to context length limit
    # input_len = 75
    num_requests = 100
    request_rate = 1
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
        num_warmups=5,
    )
    configs.append((exp_config, image_sampled_requests))

    llm_exp_config = ExperimentConfig(
        backend_config=llm_backend,
        app_id=vllm_app,
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        gpu_type=gpu_type,
        request_rate=request_rate * 10,
        image_count=0,
        use_synthesized_data=True,
        num_prompts=num_requests * 3,
        num_warmups=5,
        input_len=input_len,
        output_len=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        tokenizer_mode="auto",
        trust_remote_code=True,
    )
    llm_sampled_requests: list[SampleRequest] = VisionArenaDataset(
        dataset_path="lmarena-ai/VisionArena-Chat",
        dataset_subset=None,
        dataset_split="train",
        random_seed=48105,
    ).sample(
        num_requests=num_requests * 3,
        tokenizer=tokenizer,
        output_len=1,
        input_len=input_len,
    )
    configs.append((llm_exp_config, llm_sampled_requests))


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



