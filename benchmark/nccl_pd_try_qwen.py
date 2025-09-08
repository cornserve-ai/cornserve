import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EricConfig, ExperimentConfig, NcclPDConfig, VLLMConfig, EPDConfig, PDConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit


async def run(
    overwrite: bool = False,
) -> None:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vllm = register_app(model_id=model_id, app_type="v")
    print(f"Registered {model_id} V with ID: {vllm}")
    nccl_pd = register_app(model_id=model_id, app_type="nccl-pd")
    print(f"Registered {model_id} pd with ID: {nccl_pd}")

    vllm_config = VLLMConfig(num_replicas=2, tp_size=1)
    nccl_pd_config = NcclPDConfig(num_prefills=1, prefill_tp_size=1, num_decodes=1, decode_tp_size=1)

    configs = []
    gpu_type = "A40"
    image_width = 256
    image_height = 256
    image_count = 1
    # 1792 image tokens
    input_len = 100
    output_len = 50
    num_prompts = 300

    # InternVL3-38B # of KV cache tokens on A40 TP2
    # 72,832 -- without E
    # 96*0.9 -32*2
    # 160*0.9 -32*2
    # 72,832/3204
    # 280,000 -- without E

    # 15,632 -- with E
    # 96*0.9 -38*2
    # 160*0.9 -38*2
    # 15,632/3204
    # 105,000


    for r in [3]:
        pd_p_exp = ExperimentConfig(
            backend_config=nccl_pd_config,
            app_id=nccl_pd,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            # Dedicated prefill benchmark, so we set output_len to 1
            output_len=1,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(pd_p_exp)

    for r in [3]:
        vllm_exp = ExperimentConfig(
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
        configs.append(vllm_exp)

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
        output_data = await benchmark(request_inputs=request_inputs, config=cfg)
        completed = output_data["metrics"]["completed"]
        total_output_tokens = output_data["metrics"]["total_output"]
        if completed <= cfg.num_prompts * 0.95:
            raise RuntimeError("Insufficient completed requests")
        if total_output_tokens <= cfg.num_prompts * cfg.output_len * 0.95:
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

