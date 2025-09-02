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
    model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    # model_id: str = "OpenGVLab/InternVL3-38B"
    vllm = register_app(model_id=model_id, app_type="v")
    print(f"Registered {model_id} V with ID: {vllm}")
    pd = register_app(model_id=model_id, app_type="pd")
    print(f"Registered {model_id} pd with ID: {pd}")

    vllm_config = VLLMConfig(num_replicas=1, tp_size=2)
    # set max output tokens to 1 to profile prefill 
    pd_p_config = PDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=3, decode_tp_size=2)
    pd_d_config = PDConfig(num_prefills=3, prefill_tp_size=2, num_decodes=1, decode_tp_size=2)

    configs = []
    gpu_type = "A100"
    image_width = 1920
    image_height = 1080
    image_count = 0
    input_len = 1500
    output_len = 1000
    """
    # input_len = 2000
    # output_len = 500
    D = 1.56
    P = 3.23
    V = 1.24
    bs=96
    4 l_{epd} vs 1P3D
    4.973209471390194 vs 4.681374013977647
    4 l_{epd} vs 2P2D
    4.973209471390194 vs 3.1209160093184316

    input_len = 2000
    output_len = 1000
    4 l_{epd} vs 1P3D
    3.4206014361388095 vs 2.5888921493782093
    4 l_{epd} vs 2P2D
    3.4206014361388095 vs 1.7259280995854729
    """
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

    # we run L_{D} first bc it's the easiest part to crash due to eviction
    for r in [5]:
        pd_d_exp = ExperimentConfig(
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
        configs.append(pd_d_exp)


    for r in [5]:
        pd_p_exp = ExperimentConfig(
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
        configs.append(pd_p_exp)

    for r in [5]:
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


    if not overwrite:
        configs = [cfg for cfg in configs if not cfg.exists()]
        if not configs:
            vllm_tput = vllm_exp.load()["metrics"]["request_throughput"]
            pd_p_tput = pd_p_exp.load()["metrics"]["request_throughput"]
            pd_d_tput = pd_d_exp.load()["metrics"]["request_throughput"]

            # we only consider 8 gpu case
            print("4 l_{epd} vs 1P3D")
            print(4*vllm_tput, "vs", min(1*pd_p_tput,3*pd_d_tput), 1*pd_p_tput,3*pd_d_tput)
            print("4 l_{epd} vs 2P2D")
            print(4*vllm_tput, "vs", min(2*pd_p_tput,2*pd_d_tput), 2*pd_p_tput,2*pd_d_tput)
            exit(0)


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

    vllm_tput = vllm_exp.load()["metrics"]["request_throughput"]
    pd_p_tput = pd_p_exp.load()["metrics"]["request_throughput"]
    pd_d_tput = pd_d_exp.load()["metrics"]["request_throughput"]

    # we only consider 8 gpu case
    print("4 l_{epd} vs 1P3D")
    print(4*vllm_tput, "vs", min(1*pd_p_tput,3*pd_d_tput), 1*pd_p_tput,3*pd_d_tput)
    print("4 l_{epd} vs 2P2D")
    print(4*vllm_tput, "vs", min(2*pd_p_tput,2*pd_d_tput), 2*pd_p_tput,2*pd_d_tput)

async def main():
    set_ulimit()
    await run(overwrite=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")

