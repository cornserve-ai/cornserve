import asyncio
from typing import Literal

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EricConfig, ExperimentConfig, VLLMConfig, EPDConfig, PDConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit


async def run(
    overwrite: bool = False,
) -> None:
    # model_ids = ["OpenGVLab/InternVL3-38B", "Qwen/Qwen2.5-VL-32B-Instruct"]
    model_ids = ["OpenGVLab/InternVL3-38B"]
    # app_types: list[Literal['ev', 'v', 'e', 'epd', 'pd', 'nccl-pd']] = ["e", "ev", "v", "pd", "epd"]
    app_types: list[Literal['ev', 'v', 'e', 'epd', 'pd', 'nccl-pd']] = ["e", "ev", "v"]

    app_ids = {}
    for model_id in model_ids:
        for app_type in app_types:
            app_id = register_app(model_id=model_id, app_type=app_type)
            app_ids[(model_id, app_type)] = app_id
            print(f"Registered {model_id} {app_type} with ID: {app_id}")

    # image_width, image_height, image_count, input_len, output_len, num_prompts, rates=
    #    (e,el_l,l, pd_p, pd_d, epd_p, epd_d)
    # the request rate is different per model!
    workloads = [
        # ----- Mono is better
        (1920, 1080, 1, 100, 100, 500, (3, 3, 3, 3, 3, 3, 3)),
        (1920, 1080, 1, 100, 300, 500, (3, 2, 2, 2, 2, 2, 2)),
        # ----- EV intern wins
        (1920, 1080, 1, 1000, 300, 500, (3,2,2,2,2,2,2)),
        (1680, 1050, 1, 1000, 300, 500, (3,2,2,2,2,2,2)),
        # -----
        (1680, 1050, 1, 1000, 300, 500, (3,2,2,2,2,2,2)),
        # ----- For completeness
        (1680, 1050, 1, 100, 300, 500, (3,2,2,2,2,2,2)),
        (1680, 1050, 1, 1000, 500, 500, (2,1.2,1.2,1.2,1.2,1.2,1.2)),
        # ----- ?
        # scale up numbers
        (896, 896, 1, 100, 100, 500, (4, 6, 6, 6, 6, 6, 6)),
        (896, 896, 1, 1000, 300, 500, (4, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5)),
        (896, 896, 2, 1000, 300, 500, (2,1.2,1.2,1.2,1.2,1.2,1.2)),
        # ----- small ones
        (512, 512, 1, 100, 100, 1000, (30, 10, 10, 10, 10, 10, 10)),
        (512, 512, 1, 100, 300, 1000, (30, 8, 8, 8, 8, 8, 8)),
        # -----
    ]

    vllm_config = VLLMConfig(num_replicas=1, tp_size=2)
    # we compare single vLLM with disaggregated vLLM, ignoring Eric cost
    cornserve_l_config = CornserveConfig(num_vllms=1, vllm_tp_size=2, num_erics=6)

    # isolate Eric
    eric_config = EricConfig(num_replicas=1, tp_size=1)

    # set max output tokens to 1 to profile prefill 
    epd_p_config = EPDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=4)
    # # this might not be optimal
    # epd_d_config = EPDConfig(num_prefills=2, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=2)

    # set max output tokens to 1 to profile prefill 
    pd_p_config = PDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=3, decode_tp_size=2)
    pd_d_config = PDConfig(num_prefills=3, prefill_tp_size=2, num_decodes=1, decode_tp_size=2)

    configs = []
    gpu_type = "A100"

    tokenizers = {}
    for model_id in model_ids:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            tokenizer_mode="auto",
            trust_remote_code=True,
        )
        tokenizers[model_id] = tokenizer

    sampled_workloads = {}
    def _try_sample_workload(cfg: ExperimentConfig) -> list[SampleRequest]:
        key = (cfg.model_id, cfg.num_prompts, cfg.output_len, cfg.input_len, cfg.seed)
        if key in sampled_workloads:
            return sampled_workloads[key]
        print(f"Sampling reqeuests ...")
        tokenizer = tokenizers[cfg.model_id]
        workload: list[SampleRequest] = VisionArenaDataset(
            dataset_path="lmarena-ai/VisionArena-Chat",
            dataset_subset=None,
            dataset_split="train",
            random_seed=cfg.seed,
        ).sample(
            num_requests=cfg.num_prompts,
            tokenizer=tokenizer,
            output_len=cfg.output_len,
            input_len=cfg.input_len,
        )
        sampled_workloads[key] = workload
        return workload

    configs = []
    for model_id in model_ids:
        for workload in workloads:
            image_width, image_height, image_count, input_len, output_len, num_prompts, rates = workload
            (e_r, el_r, l_r, pd_p_r, pd_d_r, epd_p_r, epd_d_r) = rates
            for app_type in app_types:
                app_id = app_ids[(model_id, app_type)]
                if app_type == "e":
                    exp_config = ExperimentConfig(
                        backend_config=eric_config,
                        app_id=app_id,
                        model_id=model_id,
                        request_rate=e_r,
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
                elif app_type == "ev":
                    exp_config = ExperimentConfig(
                        backend_config=cornserve_l_config,
                        app_id=app_id,
                        model_id=model_id,
                        request_rate=el_r,
                        input_len=input_len,
                        output_len=output_len,
                        image_count=image_count,
                        num_prompts=num_prompts,
                        image_width=image_width,
                        image_height=image_height,
                        gpu_type=gpu_type,
                    )
                    configs.append(exp_config)
                elif app_type == "v":
                    exp_config = ExperimentConfig(
                        backend_config=vllm_config,
                        app_id=app_id,
                        model_id=model_id,
                        request_rate=l_r,
                        input_len=input_len,
                        output_len=output_len,
                        image_count=image_count,
                        num_prompts=num_prompts,
                        image_width=image_width,
                        image_height=image_height,
                        gpu_type=gpu_type,
                    )
                    configs.append(exp_config)
                elif app_type == "pd":
                    pd_p_exp_config = ExperimentConfig(
                        backend_config=pd_p_config,
                        app_id=app_id,
                        model_id=model_id,
                        request_rate=pd_p_r,
                        input_len=input_len,
                        # Dedicated prefill benchmark, so we set output_len to 1
                        output_len=1,
                        image_count=image_count,
                        num_prompts=num_prompts,
                        image_width=image_width,
                        image_height=image_height,
                        gpu_type=gpu_type,
                    )
                    configs.append(pd_p_exp_config)
                    pd_d_exp_config = ExperimentConfig(
                        backend_config=pd_d_config,
                        app_id=app_id,
                        model_id=model_id,
                        request_rate=pd_d_r,
                        input_len=input_len,
                        output_len=output_len,
                        image_count=image_count,
                        num_prompts=num_prompts,
                        image_width=image_width,
                        image_height=image_height,
                        gpu_type=gpu_type,
                    )
                    configs.append(pd_d_exp_config)
                elif app_type == "epd":
                    epd_p_exp_config = ExperimentConfig(
                        backend_config=epd_p_config,
                        app_id=app_id,
                        model_id=model_id,
                        request_rate=epd_p_r,
                        input_len=input_len,
                        # Dedicated prefill benchmark, so we set output_len to 1
                        output_len=1,
                        image_count=image_count,
                        num_prompts=num_prompts,
                        image_width=image_width,
                        image_height=image_height,
                        gpu_type=gpu_type,
                    )
                    configs.append(epd_p_exp_config)
                else:
                    raise NotImplementedError(f"Backend {app_type} is not supported.")

    if not overwrite:
        configs = [cfg for cfg in configs if not cfg.exists()]

    # prioritize by request rate
    configs.sort(key=lambda config: (-config.request_rate,))

    print(f"Total configs: {len(configs)}")

    for i, cfg in enumerate(configs):
        print(f"Current {i}/{len(configs)} config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        if cfg.exists():
            # there are overlapping configs
            print(f"Config already exists, skipping ...")
            continue
        # we scale every time to clean up the task executors states just in case
        sampled_requests = _try_sample_workload(cfg)
        await scale(cfg)
        request_inputs = transform_sampled_requests(config=cfg, sampled_requests=sampled_requests)
        output_data = await benchmark(request_inputs=request_inputs, config=cfg)
        completed = output_data["metrics"]["completed"]
        total_output_tokens = output_data["metrics"]["total_output"]
        if completed <= cfg.num_prompts * 0.95:
            raise RuntimeError("Insufficient completed requests")
        if not isinstance(cfg.backend_config, EricConfig):
            if total_output_tokens <= sum(r.expected_output_len for r in sampled_requests) * 0.95:
                raise RuntimeError("Insufficient output tokens")
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

