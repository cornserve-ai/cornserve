import asyncio

from cornserve.utils import set_ulimit
from transformers import AutoTokenizer

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, ExperimentConfig, vLLMConfig

async def run_qwen2_5_vl_32b(
    overwrite: bool = False,
) -> None:
    model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    ev = register_app(model_id=model_id, app_type="ev")
    print(f"Registered Qwen2.5-VL-32B EV with ID: {ev}")
    e = register_app(model_id=model_id, app_type="e")
    print(f"Registered Qwen2.5-VL-32B E with ID: {e}")
    vllm = register_app(model_id=model_id, app_type="v")
    print(f"Registered Qwen2.5-VL-32B V with ID: {vllm}")

    vllm_config = vLLMConfig(num_replicas=4, tp_size=2)
    cornserve_config = CornserveConfig(num_vllms=3, vllm_tp_size=2, num_erics=2)

    configs = []
    image_width = 1280
    image_height = 720
    input_len = 300
    output_len = 300
    image_count = 1
    num_prompts = 1000
    for r in []:
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
        )
        configs.append(exp_config)
    for r in [5]:
        exp_config = ExperimentConfig(
            backend_config=cornserve_config,
            app_id=ev,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
        )
        configs.append(exp_config)

    if not overwrite:
        configs = [cfg for cfg in configs if not cfg.exists()]

    # prioritize by request rate
    configs.sort(key=lambda config: (-config.request_rate, ))

    print(f"Total configs: {len(configs)}")

    shared_config = next(iter(configs))
    tokenizer = AutoTokenizer.from_pretrained(
        shared_config.model_id,
        tokenizer_mode = "auto",
        trust_remote_code=True,
    )

    print(f"Sampling reqeuests ...")
    sampled_requests: list[SampleRequest]= VisionArenaDataset(
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
    await run_qwen2_5_vl_32b()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
