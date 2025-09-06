"""Execute throughput benchmark for InternVL3-38B model."""

import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EPDConfig, ExperimentConfig, PDConfig, VLLMConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit


async def run(
    overwrite: bool = False,
) -> None:
    """Run throughput benchmark for InternVL3-38B model."""
    model_id: str = "OpenGVLab/InternVL3-38B"
    ev = register_app(model_id=model_id, app_type="ev")
    print(f"Registered {model_id} EV with ID: {ev}")
    vllm = register_app(model_id=model_id, app_type="v")
    print(f"Registered {model_id} V with ID: {vllm}")
    pd = register_app(model_id=model_id, app_type="pd")
    print(f"Registered {model_id} pd with ID: {pd}")
    epd = register_app(model_id=model_id, app_type="epd")
    print(f"Registered {model_id} epd with ID: {epd}")

    vllm_config = VLLMConfig(num_replicas=4, tp_size=2)
    cornserve_config = CornserveConfig(num_vllms=3, vllm_tp_size=2, num_erics=2)
    pd_config = PDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=3, decode_tp_size=2)
    pd_config2 = PDConfig(num_prefills=2, prefill_tp_size=2, num_decodes=2, decode_tp_size=2)
    pd_config3 = PDConfig(num_prefills=3, prefill_tp_size=2, num_decodes=1, decode_tp_size=2)
    epd_config = EPDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=2, decode_tp_size=2, num_erics=2)
    epd_config2 = EPDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=4)
    epd_config3 = EPDConfig(num_prefills=2, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=2)
    gpu_type = "A100"

    configs = []
    """
    image_width = 1920
    image_height = 1080
    # 1920*1080 --> 2304
    # 1680*1050 --> 1796
    input_len = 900
    output_len = 400
    image_count = 1
    num_prompts = 2000
    # 2304+900+400 = 3604
    # 3604 --> bs = 74, 94
    image_width = 1920
    image_height = 1080
    # 1920*1080 --> 2304
    # 1680*1050 --> 1796
    input_len = 800
    output_len = 400
    image_count = 1
    num_prompts = 2000
    # 270432, 344032
    # 2304+800+400 = 3504
    # 3504 --> bs = 76, 97
    """

    # 1920*1080 --> 2304
    # 1680*1050 --> 1796
    image_width = 1920
    image_height = 1080
    input_len = 1000
    output_len = 300
    image_count = 1
    num_prompts = 2000
    # max bs not controlled

    # 270432, 344032
    # 2304+800+400 = 3554
    # 3504 --> bs = 75, 95

    # 2304+1000+300 = 3604
    # 2304+900+400 = 3604
    # 3604 --> bs = 74, 94

    # 1792 + 1400 + 300 = 3600
    # bs: 71, 90
    # image_width = 1680
    # image_height = 1050
    # input_len = 100
    # output_len = 50
    # image_count = 2
    # num_prompts = 2000

    # 1792 * 2 + 150 = 3750
    # bs: 71, 90
    # image_width = 1680
    # image_height = 1050
    # input_len = 100
    # output_len = 50
    # image_count = 2
    # num_prompts = 2000
    # cornserve_config = CornserveConfig(num_vllms=2, vllm_tp_size=2, num_erics=4)

    request_rates = [3.5]
    # FINAL 3.5!!! No BS limit
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
            gpu_type = gpu_type,
        )
        configs.append(exp_config)
    for r in request_rates:
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
            gpu_type = gpu_type,
        )
        configs.append(exp_config)

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
            gpu_type = gpu_type,
        )
        configs.append(exp_config)
        exp_config = ExperimentConfig(
            backend_config=pd_config2,
            app_id=pd,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type = gpu_type,
        )
        configs.append(exp_config)
        exp_config = ExperimentConfig(
            backend_config=pd_config3,
            app_id=pd,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type = gpu_type,
        )
        configs.append(exp_config)

    for r in request_rates:
        exp_config = ExperimentConfig(
            backend_config=epd_config,
            app_id=epd,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type = gpu_type,
        )
        configs.append(exp_config)
        exp_config = ExperimentConfig(
            backend_config=epd_config2,
            app_id=epd,
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type = gpu_type,
        )
        configs.append(exp_config)
        exp_config = ExperimentConfig(
            backend_config=epd_config3,
            app_id=epd,
            model_id=model_id,
            request_rate=r,
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
