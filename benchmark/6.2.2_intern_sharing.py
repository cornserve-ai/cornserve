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
    if overwrite:
        print("WARNING!!!! Overwrite mode is enabled. Existing configurations will be re-evaluated.")

    model_ids = ["OpenGVLab/InternVL3-38B"]
    app_types: list[Literal['ev', 'v']] = ["ev", "v"]

    app_ids = {}
    for model_id in model_ids:
        for app_type in app_types:
            app_id = register_app(model_id=model_id, app_type=app_type)
            app_ids[(model_id, app_type)] = app_id
            print(f"Registered {model_id} {app_type} with ID: {app_id}")

    image_probability = 1

    # image_width, image_height, image_count, input_len, output_len, num_prompts, rates, encoder_fission_probability
    #    (el, l, pd, epd)
    # the request rate is different per model!
    workloads = [
        (1680, 1050, 1, 100, 50, 1500, (7, 7, 7, 7), 0.7),
        (1920, 1080, 1, 100, 50, 1500, (7, 7, 7, 7), 0.7),

        # (1680, 1050, 2, 100, 50, 4000, (7, 7, 7, 7), 0.6),
        # (1680, 1050, 2, 100, 50, 4000, (7, 7, 7, 7), 0.7),

        # (1920, 1080, 2, 100, 150, 3000, (6.25, 6.25, 6.25, 6.25), 0.75),
        # (1920, 1080, 2, 100, 300, 2000, (3, 3, 3, 3), 0.775),
        # (1920, 1080, 2, 100, 300, 2000, (3, 3, 3, 3), 0.75),

        # (1680, 1050, 4, 100, 100, 2000, (6.25, 6.25, 6.25, 6.25), 0.675),

        # 0.5 => too little work for Eric
        # 0.6875 => Eric bottlenecs a little bit, but iternode show imbalance
        # 1.8081
        # (1920, 1080, 4, 100, 50, 2000, (2.5, 2.5, 2.5, 2.5), 0.6875),

        # (1920, 1080, 4, 100, 100, 2000, (2.5, 2.5, 2.5, 2.5), 0.5),
        # (1920, 1080, 8, 100, 50, 500, (3, 3, 3, 3)),
        #
        # (2560, 1440, 1, 100, 50, 500, (3, 3, 3, 3)),
        # (2560, 1440, 2, 100, 50, 500, (3, 3, 3, 3)),
        # (2560, 1440, 4, 100, 50, 500, (3, 3, 3, 3)),
        # (2560, 1440, 8, 100, 50, 500, (3, 3, 3, 3)),
        #
        # (4032, 3024, 1, 100, 50, 500, (3, 3, 3, 3)),
        # (4032, 3024, 2, 100, 50, 500, (3, 3, 3, 3)),
    ]

    vllm_config = VLLMConfig(num_replicas=8, tp_size=2)
    # we compare single vLLM with disaggregated vLLM, ignoring Eric cost
    cornserve_config = CornserveConfig(num_vllms=7, vllm_tp_size=2, num_erics=2)

    # set max output tokens to 1 to profile prefill 
    epd_config = EPDConfig(num_prefills=1, prefill_tp_size=1, num_decodes=1, decode_tp_size=1, num_erics=4)

    # set max output tokens to 1 to profile prefill 
    pd_config = PDConfig(num_prefills=1, prefill_tp_size=1, num_decodes=3, decode_tp_size=1)

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
            image_width, image_height, image_count, input_len, output_len, num_prompts, rates, encoder_fission_probability = workload
            (el, l, pd, epd) = rates
            for app_type in app_types:
                app_id = app_ids[(model_id, app_type)]
                if app_type == "ev":
                    backend_config = cornserve_config
                    request_rate = el
                elif app_type == "v":
                    backend_config = vllm_config
                    request_rate = l
                elif app_type == "pd":
                    backend_config = pd_config
                    request_rate = pd
                elif app_type == "epd":
                    backend_config = epd_config
                    request_rate = epd
                else:
                    raise NotImplementedError(f"Backend {app_type} is not supported.")
                exp_config = ExperimentConfig(
                    backend_config=backend_config,
                    app_id=app_id,
                    model_id=model_id,
                    request_rate=request_rate,
                    input_len=input_len,
                    output_len=output_len,
                    image_count=image_count,
                    num_prompts=num_prompts,
                    image_width=image_width,
                    image_height=image_height,
                    image_probability=image_probability,
                    gpu_type=gpu_type,
                    encoder_fission_probability=encoder_fission_probability if app_type in ("ev", ) else 1,
                )
                configs.append(exp_config)

    if not overwrite:
        configs = [cfg for cfg in configs if not cfg.exists()]

    # prioritize by request rate
    configs.sort(key=lambda config: (-config.request_rate,))

    print(f"Total configs: {len(configs)}")

    for i, cfg in enumerate(configs):
        print(f"Current {i}/{len(configs)} config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        print(cfg)
        if cfg.exists() and not overwrite:
            # there are overlapping configs
            print(f"Config already exists, skipping ...")
            continue
        # we scale every time to clean up the task executors states just in case
        sampled_requests = _try_sample_workload(cfg)
        # await scale(cfg)
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

