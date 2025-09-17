import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, ServeGenDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EricConfig, ExperimentConfig, HFOmniConfig, OmniServeGenConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit

async def run(
    overwrite: bool = False,
) -> None:
    model_id = "Qwen/Qwen2.5-Omni-7B"

    N=16
    hf_app = register_app(model_id=model_id, app_type="hf-omni")
    hf_omni_config = HFOmniConfig(num_replicas=N)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )

    # rquest rate, duration, no_image_prob, video_prob, audio_prob, return_audio_prob
    worloads = [
        (1, 300, 0.0, 0.5, 0.5, 0.2),
    ]

    gpu_type = "A100"

    sampled_workloads = {}
    def _try_sample_workload(
        seed: int,
        serve_gen_config: OmniServeGenConfig,
    ) -> list[SampleRequest]:
        key = (
            serve_gen_config.request_rate,
            serve_gen_config.duration,
            serve_gen_config.no_image_prob,
            serve_gen_config.audio_prob,
            serve_gen_config.video_prob,
            serve_gen_config.return_audio_prob,
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
            return_audio_prob=serve_gen_config.return_audio_prob,
        )
        sampled_workloads[key] = workload
        return workload

    configs = []

    for workload in worloads:
        request_rate, duration, no_image_prob, video_prob, audio_prob, return_audio_prob = workload
        servegen_omni_workload_config = OmniServeGenConfig(
            request_rate=request_rate,
            duration=duration,
            no_image_prob=no_image_prob,
            video_prob=video_prob,
            audio_prob=audio_prob,
            return_audio_prob=return_audio_prob,
        )
        exp_config = ExperimentConfig(
            backend_config=hf_omni_config,
            app_id=hf_app,
            model_id=model_id,
            gpu_type=gpu_type,
            dataset="servegen",
            workload_config=servegen_omni_workload_config,
            request_rate=request_rate,
            use_synthesized_data=False,
        )
        sampled_requests = _try_sample_workload(
            seed=exp_config.seed,
            serve_gen_config=servegen_omni_workload_config,
        )
        configs.append((exp_config, sampled_requests))

    # prioritize by request rate
    configs.sort(key=lambda config: (-config[0].request_rate,))
    configs = configs if overwrite else [cfg for cfg in configs if not cfg[0].exists()]

    print(f"Total configs: {len(configs)}")

    for cfg, sampled_requests in configs:
        print(f"Current config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        # we scale every time to clean up the task executors states just in case
        await scale(cfg)
        request_inputs = transform_sampled_requests(config=cfg, sampled_requests=sampled_requests)
        output_data = await benchmark(request_inputs=request_inputs, config=cfg)
        completed = output_data["metrics"]["completed"]
        if completed <= len(sampled_requests) * 0.95:
            raise RuntimeError("Insufficient completed requests")
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
