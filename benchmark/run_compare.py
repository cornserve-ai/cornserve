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
    eric_batch_dize = 1
    # model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_id: str = "OpenGVLab/InternVL3-38B"
    ev = register_app(model_id=model_id, app_type="ev")
    print(f"Registered {model_id} EV with ID: {ev}")
    e = register_app(model_id=model_id, app_type="e", eric_max_batch_size=eric_batch_dize)
    print(f"Registered {model_id} E with ID: {e}")
    vllm = register_app(model_id=model_id, app_type="v")
    print(f"Registered {model_id} V with ID: {vllm}")
    epd = register_app(model_id=model_id, app_type="epd")
    print(f"Registered {model_id} epd with ID: {epd}")
    pd = register_app(model_id=model_id, app_type="pd")
    print(f"Registered {model_id} pd with ID: {pd}")

    vllm_config = VLLMConfig(num_replicas=1, tp_size=2)
    # we compare single vLLM with disaggregated vLLM, ignoring Eric cost
    cornserve_l_config = CornserveConfig(num_vllms=1, vllm_tp_size=2, num_erics=6)

    # isolate Eric
    eric_config = EricConfig(num_replicas=1, tp_size=1, max_batch_size=eric_batch_dize)

    # set max output tokens to 1 to profile prefill 
    epd_p_config = EPDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=4)
    # this might not be optimal
    epd_d_config = EPDConfig(num_prefills=2, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=2)

    # set max output tokens to 1 to profile prefill 
    pd_p_config = PDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=3, decode_tp_size=2)
    pd_d_config = PDConfig(num_prefills=3, prefill_tp_size=2, num_decodes=1, decode_tp_size=2)

    configs = []
    gpu_type = "A100"
    image_width = 1920
    image_height = 1080
    image_count = 1
    input_len = 100
    output_len = 300
    num_prompts = 500

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

    for r in [10]:
        exp_config = ExperimentConfig(
            backend_config=eric_config,
            app_id=e,
            model_id=model_id,
            request_rate=r,
            # Dedicated Eric profile
            input_len=0,
            output_len=0,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(exp_config)

    # we run L_{D} first bc it's the easiest part to crash due to eviction
    for r in [5]:
        exp_config = ExperimentConfig(
            backend_config=pd_d_config,
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

    # then we run L_{PD} bc CUDA IPC somehow always use GPU 0 (bug?) and having high GPU utilization
    # may cause CUDA OOM
    for r in [5]:
        exp_config = ExperimentConfig(
            backend_config=cornserve_l_config,
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
        configs.append(exp_config)


    for r in [5]:
        exp_config = ExperimentConfig(
            backend_config=pd_p_config,
            app_id=pd,
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
        configs.append(exp_config)

    for r in [5]:
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

    for r in [5]:
        exp_config = ExperimentConfig(
            backend_config=epd_p_config,
            app_id=epd,
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
        configs.append(exp_config)

    for r in [5]:
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

