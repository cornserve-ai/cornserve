import asyncio
from typing import Literal

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, ServeGenDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, ExperimentConfig, ServeGenConfig, VLLMConfig, EPDConfig, PDConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit

async def run(
    overwrite: bool = False,
) -> None:
    # model_ids = ["OpenGVLab/InternVL3-38B", "Qwen/Qwen2.5-VL-32B-Instruct"]
    model_id = "OpenGVLab/InternVL3-38B"
    # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    # request_rates = [3,3.5,4]

    app_types: list[Literal['ev', 'v', 'e', 'epd', 'pd', 'nccl-pd']] = ["ev", "v"]
    request_rates = [2.65]
    duration = 900 # 15 minutes
    seed = 48105

    app_ids = {}
    for app_type in app_types:
        app_id = register_app(model_id=model_id, app_type=app_type)
        app_ids[(model_id, app_type)] = app_id
        print(f"Registered {model_id} {app_type} with ID: {app_id}")

    backend_configs = []
    vllm_config = VLLMConfig(num_replicas=4, tp_size=2)
    backend_configs.append(vllm_config)
    for num_vllms in [3]:
        num_erics = (8 // 2 - num_vllms) * 2
        cornserve_config = CornserveConfig(num_vllms=num_vllms, vllm_tp_size=2, num_erics=num_erics)
        backend_configs.append(cornserve_config)

    # request rate -> sampled_requests
    sampled_workloads = {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )
    def _try_sample_workload(serve_gen_config: ServeGenConfig) -> list[SampleRequest]:
        key = (
            serve_gen_config.request_rate,
            serve_gen_config.duration,
            serve_gen_config.no_image_prob,
            serve_gen_config.audio_prob,
            serve_gen_config.video_prob,
        )
        if key in sampled_workloads:
            return sampled_workloads[key]
        print(f"Sampling reqeuests ...")
        workload: list[SampleRequest] = ServeGenDataset(seed).sample(
            tokenizer=tokenizer,
            request_rate=serve_gen_config.request_rate,
            duration=serve_gen_config.duration,
            no_image_prob=serve_gen_config.no_image_prob,
            audio_prob=serve_gen_config.audio_prob,
            video_prob=serve_gen_config.video_prob,
        )
        sampled_workloads[key] = workload
        return workload

    gpu_type = "A100"
    configs = []
    for request_rate in request_rates:
        serve_gen_config = ServeGenConfig(
            request_rate=request_rate,
            duration=duration,
            no_image_prob=0.2,
        )
        for backend in backend_configs:
            if isinstance(backend, CornserveConfig):
                app_type = "ev"
            elif isinstance(backend, VLLMConfig):
                app_type = "v"
            else:
                raise NotImplementedError(f"Backend {backend} is not supported.")
            if app_type not in app_types:
                continue
            app_id = app_ids[(model_id, app_type)]
            exp_config = ExperimentConfig(
                backend_config=backend,
                app_id=app_id,
                model_id=model_id,
                gpu_type = gpu_type,
                dataset="servegen",
                seed=seed,
                # 
                workload_config=serve_gen_config,
                request_rate=request_rate,
                # !!!! this must be set
                use_synthesized_data=False,
                # use default args for the rest
            )
            configs.append(exp_config)

    # prioritize by request rate
    configs.sort(key=lambda config: (-config.request_rate,))
    configs = configs if overwrite else [cfg for cfg in configs if not cfg.exists()]

    print(f"Total configs: {len(configs)}")

    for cfg in configs:
        print(f"Current config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        serve_gen_rr = cfg.workload_config.request_rate
        sampled_requests = _try_sample_workload(cfg.workload_config)
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
    await run(overwrite=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")

