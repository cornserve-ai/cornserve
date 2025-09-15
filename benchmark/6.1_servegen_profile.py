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
    # model_ids = ["OpenGVLab/InternVL3-38B", "Qwen/Qwen2.5-VL-32B-Instruct"]
    model_ids = ["Qwen/Qwen2.5-VL-32B-Instruct"]
    # app_types: list[Literal['ev', 'v', 'e', 'epd', 'pd', 'nccl-pd']] = ["e", "ev", "v", "pd", "epd"]
    app_types: list[Literal['ev', 'v', 'e', 'epd', 'pd', 'nccl-pd']] = [
        "e",
        "v",
        "ev",
    ]
    app_modalities = [
        [["IMAGE"], ["VIDEO"]],  # e
        [["VIDEO", "IMAGE"]], # v
        [["VIDEO", "IMAGE"]], # ev
    ]
    assert len(app_types) == len(app_modalities)

    app_ids = {}
    for model_id in model_ids:
        for i, app_type in enumerate(app_types):
            modlities_list = app_modalities[i]
            for modalities in modlities_list:
                print(f"Registering {model_id} {app_type} with modalities {modalities} ...")
                app_id = register_app(
                    model_id=model_id,
                    app_type=app_type,
                    modalities=modalities,  # type: ignore
                )
                app_ids[(model_id, app_type, tuple(modalities))] = app_id
                print(f"Registered {model_id} {app_type} with ID: {app_id}")
    print(app_ids)

    # rates, video_prob, duration, image_fission_probability, video_fission_probability
    #    (el, l, e_img, e_vid, pd, epd)
    # the request rate is different per model!
    workloads = [
        # one image is not enough to trigger sharing
        # when video prob is low, we need a longer duration for steady state
        ((0.375, 0.5, 5, 5, 0.375, 0.375), 0.5, 360, 1.0, 1.0),
    ]

    vllm_config = VLLMConfig(num_replicas=1, tp_size=2)
    # we compare single vLLM with disaggregated vLLM, ignoring Eric cost
    cornserve_l_config = CornserveConfig(
        num_vllms=1,
        vllm_tp_size=2,
        num_erics=6,
        num_video_erics=4,
        num_image_erics=2,
        modalities=["image", "video"]
    )
    # isolate Eric
    img_eric_config = EricConfig(num_replicas=1, tp_size=1, modality="image")
    vid_eric_config = EricConfig(num_replicas=1, tp_size=1, modality="video")

    # set max output tokens to 1 to profile prefill 
    epd_p_config = EPDConfig(num_prefills=1, prefill_tp_size=1, num_decodes=1, decode_tp_size=1, num_erics=4)
    # # this might not be optimal
    # epd_d_config = EPDConfig(num_prefills=2, prefill_tp_size=1, num_decodes=1, decode_tp_size=1, num_erics=2)

    # set max output tokens to 1 to profile prefill 
    pd_p_config = PDConfig(num_prefills=1, prefill_tp_size=1, num_decodes=3, decode_tp_size=1)
    pd_d_config = PDConfig(num_prefills=3, prefill_tp_size=1, num_decodes=1, decode_tp_size=1)

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

    def build_serve_gen_config(
        request_rate: float,
        duration: int,
        video_prob: float,
    ) -> ServeGenConfig:
        return ServeGenConfig(
            request_rate=request_rate,
            duration=duration,
            no_image_prob=0.0,
            audio_prob=0.0,
            video_prob=video_prob,
        )

    configs = []
    for model_id in model_ids:
        for workload in workloads:
            rates, video_prob, duration, image_fission_probability, video_fission_probability = workload
            (el_r, l_r, e_img_r, e_vid_r, pd_r, epd_r) = rates
            for i, app_type in enumerate(app_types):
                modlities_list = app_modalities[i]
                for modalities in modlities_list:
                    app_id = app_ids[(model_id, app_type, tuple(modalities))]
                    if app_type == "e" and modalities == ["IMAGE"]:
                        serve_gen_config = build_serve_gen_config(request_rate=e_img_r, duration=duration, video_prob=0)
                        exp_config = ExperimentConfig(
                            backend_config=img_eric_config,
                            app_id=app_id,
                            model_id=model_id,
                            request_rate=e_img_r,
                            # Dedicated Eric profile
                            input_len=0,
                            output_len=0,
                            gpu_type=gpu_type,
                            dataset="servegen",
                            workload_config=serve_gen_config,
                            use_synthesized_data=False,
                        )
                        configs.append(exp_config)
                    elif app_type == "e" and modalities == ["VIDEO"]:
                        serve_gen_config = build_serve_gen_config(request_rate=e_vid_r, duration=duration, video_prob=video_prob)
                        exp_config = ExperimentConfig(
                            backend_config=vid_eric_config,
                            app_id=app_id,
                            model_id=model_id,
                            request_rate=e_vid_r,
                            # Dedicated Eric profile
                            input_len=0,
                            output_len=0,
                            gpu_type=gpu_type,
                            dataset="servegen",
                            workload_config=serve_gen_config,
                            use_synthesized_data=False,
                        )
                        configs.append(exp_config)
                    elif app_type == "ev":
                        serve_gen_config = build_serve_gen_config(request_rate=el_r, duration=duration, video_prob=video_prob)
                        exp_config = ExperimentConfig(
                            backend_config=cornserve_l_config,
                            app_id=app_id,
                            model_id=model_id,
                            request_rate=el_r,
                            gpu_type=gpu_type,
                            image_fission_probability=image_fission_probability,
                            video_fission_probability=video_fission_probability,
                            dataset="servegen",
                            workload_config=serve_gen_config,
                            use_synthesized_data=False,
                        )
                        configs.append(exp_config)
                    elif app_type == "v":
                        serve_gen_config = build_serve_gen_config(request_rate=l_r, duration=duration, video_prob=video_prob)
                        exp_config = ExperimentConfig(
                            backend_config=vllm_config,
                            app_id=app_id,
                            model_id=model_id,
                            request_rate=l_r,
                            gpu_type=gpu_type,
                            dataset="servegen",
                            workload_config=serve_gen_config,
                            use_synthesized_data=False,
                        )
                        configs.append(exp_config)
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

