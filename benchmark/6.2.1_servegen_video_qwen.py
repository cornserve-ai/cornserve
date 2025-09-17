import asyncio
from typing import Literal

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, ServeGenDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EricConfig, ExperimentConfig, VLLMConfig, EPDConfig, PDConfig, ServeGenConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit

async def run(
    overwrite: bool = False,
) -> None:
    if overwrite:
        print("WARNING!!!! Overwrite mode is enabled. Existing configurations will be re-evaluated.")

    model_ids = ["Qwen/Qwen2.5-VL-32B-Instruct"]
    app_types: list[Literal['ev', 'v', "pd", "epd"]] = ["pd", "epd"]

    app_ids = {}
    for model_id in model_ids:
        for app_type in app_types:
            app_id = register_app(
                model_id=model_id,
                app_type=app_type,
                modalities=["VIDEO", "IMAGE"],
            )
            app_ids[(model_id, app_type)] = app_id
            print(f"Registered {model_id} {app_type} with ID: {app_id}")

    # rates, video_prob, duration, image_fission_prob, video_fission_probability
    #    (el, l, pd, epd)
    # the request rate is different per model!
    workloads = [
        # one image is not enough to trigger sharing
        # when video prob is low, we need a longer duration for steady state
        # ((4, 4, 0.375, 0.375), 0.5, 600, 1, 0.875),
        # ((4, 4, 0.375, 0.375), 0.5, 600, 1, 0.875),

        # ((4, 4, 4, 4), 0.5, 600, 1, 0.5),
        # ((3.5, 3.5, 3.5, 3.5), 1, 600, 0, 0.6),
        # ((2.15, 2.15, 2.15, 2.15), 1, 600, 0, 0.85),

        # ((2.55, 2.55, 2.55, 2.55), 0.5, 600, 0, 1),
        ((1.25, 1.25, 0.6, 0.6), 0.5, 600, 0, 1),

        # ((4, 4, 4, 4), 0.5, 600, 1, 0.875),

        # ((1.5, 1.5, 1.5, 1.5), 0.5, 720, 1, 0.875),
        # ((5, 5, 5, 5), 0.7, 1200, 0.85),
    ]

    vllm_config = VLLMConfig(num_replicas=4, tp_size=2)
    # we compare single vLLM with disaggregated vLLM, ignoring Eric cost
    cornserve_config = CornserveConfig(
        num_vllms=7,
        vllm_tp_size=2,
        num_erics=2,
        num_video_erics=2,
        num_image_erics=0,
        modalities=["image", "video"]
    )

    # set max output tokens to 1 to profile prefill 
    pd_config = PDConfig(num_prefills=2, prefill_tp_size=2, num_decodes=2, decode_tp_size=2)
    epd_config = EPDConfig(num_prefills=1, prefill_tp_size=1, num_decodes=2, decode_tp_size=2, num_erics=2, num_image_erics=1, num_video_erics=1)

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
        serve_gen_config = cfg.workload_config
        assert isinstance(serve_gen_config, ServeGenConfig)
        key = (
            cfg.model_id,
            cfg.seed,
            serve_gen_config.request_rate,
            serve_gen_config.duration,
            serve_gen_config.no_image_prob,
            serve_gen_config.audio_prob,
            serve_gen_config.video_prob,
        )
        if key in sampled_workloads:
            return sampled_workloads[key]
        print(f"Sampling reqeuests ...")
        tokenizer = tokenizers[cfg.model_id]
        workload: list[SampleRequest] = ServeGenDataset(cfg.seed).sample(
            tokenizer=tokenizer,
            request_rate=serve_gen_config.request_rate,
            duration=serve_gen_config.duration,
            no_image_prob=serve_gen_config.no_image_prob,
            audio_prob=serve_gen_config.audio_prob,
            video_prob=serve_gen_config.video_prob,
        )
        sampled_workloads[key] = workload
        return workload

    configs = []
    for model_id in model_ids:
        for workload in workloads:
            rates, video_prob, duration, image_fission_probability, video_fission_probability = workload
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
                serve_gen_config = ServeGenConfig(
                    request_rate=request_rate,
                    duration=duration,
                    no_image_prob=0.0,
                    audio_prob=0.0,
                    video_prob=video_prob,
                )
                exp_config = ExperimentConfig(
                    backend_config=backend_config,
                    app_id=app_id,
                    model_id=model_id,
                    request_rate=request_rate,
                    gpu_type=gpu_type,
                    dataset="servegen",
                    # encoder_fission_probability=encoder_fission_probability if app_type in ("ev", ) else 1,
                    image_fission_probability=image_fission_probability if app_type in ("ev", ) else 1,
                    video_fission_probability=video_fission_probability if app_type in ("ev", ) else 1,
                    workload_config=serve_gen_config,
                    use_synthesized_data=False,
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
        request_inputs = transform_sampled_requests(config=cfg, sampled_requests=sampled_requests)
        await scale(cfg)
        output_data = await benchmark(request_inputs=request_inputs, config=cfg)
        completed = output_data["metrics"]["completed"]
        total_output_tokens = output_data["metrics"]["total_output"]
        if completed <= len(sampled_requests) * 0.95:
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

