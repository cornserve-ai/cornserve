import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, ServeGenDataset, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, ExperimentConfig, ServeGenConfig, VLLMConfig, EPDConfig, PDConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit

async def run(
    overwrite: bool = False,
) -> None:
    # model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_id: str = "Qwen/Qwen2.5-Omni-7B"
    vllm = register_app(
        model_id=model_id,
        modalities=["AUDIO", "IMAGE", "VIDEO"],
        app_type="v",
    )
    print(f"Registered {model_id} V with ID: {vllm}")
    vllm_config = VLLMConfig(num_replicas=1, tp_size=1)

    configs = []
    gpu_type = "A40"

    serve_gen_config = ServeGenConfig(
        request_rate=2.0,
        duration=300,
        no_image_prob=0.3,
        audio_prob=0.3,
        video_prob=0.3,
    )

    for r in [2]:
        vllm_exp = ExperimentConfig(
            backend_config=vllm_config,
            app_id=vllm,
            model_id=model_id,
            request_rate=r,
            gpu_type=gpu_type,
            dataset="servegen",
            workload_config=serve_gen_config,
            # !!!! this must be set
            use_synthesized_data=False,
        )
        configs.append(vllm_exp)

    # prioritize by request rate
    configs.sort(key=lambda config: (-config.request_rate,))
    configs = configs if overwrite else [cfg for cfg in configs if not cfg.exists()]

    print(f"Total configs: {len(configs)}")

    shared_config = next(iter(configs))
    tokenizer = AutoTokenizer.from_pretrained(
        shared_config.model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )

    print(f"Sampling reqeuests ...")
    sampled_requests: list[SampleRequest] = ServeGenDataset().sample(
        tokenizer=tokenizer,
        request_rate=serve_gen_config.request_rate,
        duration=serve_gen_config.duration,
        no_image_prob=serve_gen_config.no_image_prob,
        audio_prob=serve_gen_config.audio_prob,
        video_prob=serve_gen_config.video_prob,
    )

    for cfg in configs:
        print(f"Current config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        # we scale every time to clean up the task executors states just in case
        await scale(cfg)
        request_inputs = transform_sampled_requests(config=cfg, sampled_requests=sampled_requests)
        output_data = await benchmark(request_inputs=request_inputs, config=cfg)
        completed = output_data["metrics"]["completed"]
        total_output_tokens = output_data["metrics"]["total_output"]
        if completed <= len(sampled_requests) * 0.95:
            raise RuntimeError("Insufficient completed requests")
        if total_output_tokens <= sum(r.expected_output_len for r in sampled_requests) * 0.95:
            raise RuntimeError("Insufficient output tokens")
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


