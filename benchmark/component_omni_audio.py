import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, SyntheticOmniDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EricConfig, ExperimentConfig, OmniConfig, VLLMConfig, VLLMOmniConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit

async def run(
    overwrite: bool = False,
) -> None:
    model_id = "Qwen/Qwen2.5-Omni-7B"

    omni = register_app(
        model_id=model_id,
        app_type="la",
        modalities=["IMAGE", "VIDEO", "AUDIO"],
    )

    omni_config = VLLMOmniConfig(
        num_thinkers=7,
        num_talker_vocoders=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )

    gpu_type = "A100"
    num_requests = 200
    audio_gen_request_rate = 0.5
    input_len = 1000
    # make it short for audio output
    output_len = 30
    image_width = 800
    image_height = 600
    # 609 image tokens
    video_num_frames = 20
    video_width = 1024
    video_height = 768
    # 9990 video tokens
    audio_duration_sec = 8
    # 200 audio tokens

    llm_workload_config = OmniConfig(
        include_image=True,
        include_audio=True,
        include_video=True,
        image_width=image_width,
        image_height=image_height,
        video_num_frames=video_num_frames,
        video_height=video_height,
        video_width=video_width,
        audio_duration_sec=audio_duration_sec,
        return_audio_prob=1.0,
    )

    def smaple_requests(
        num_requests: int,
        config: OmniConfig,
        input_len: int = 0,
        output_len: int = 0,
        return_audio_prob: float = 0.0,
    ) -> list[SampleRequest]:
        return SyntheticOmniDataset().sample(
            num_requests=num_requests,
            tokenizer=tokenizer,
            input_len=input_len,
            output_len=output_len,
            include_image=config.include_image,
            image_width=config.image_width,
            image_height=config.image_height,
            include_video=config.include_video,
            video_num_frames=config.video_num_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            include_audio=config.include_audio,
            audio_duration_sec=config.audio_duration_sec,
            return_audio_prob=return_audio_prob,
        )

    llm_sampled_requests = smaple_requests(
        num_requests,
        llm_workload_config,
        input_len=input_len,
        output_len=output_len,
        return_audio_prob=llm_workload_config.return_audio_prob,
    )

    configs = []

    exp_config = ExperimentConfig(
        backend_config=omni_config,
        app_id=omni,
        model_id=model_id,
        gpu_type=gpu_type,
        dataset="omni",
        workload_config=llm_workload_config,
        request_rate=audio_gen_request_rate,
        use_synthesized_data=False,
        num_warmups=0,
    )
    configs.append((exp_config, llm_sampled_requests))

    # prioritize by request rate
    configs.sort(key=lambda config: (-config[0].request_rate,))
    configs = configs if overwrite else [cfg for cfg in configs if not cfg[0].exists()]

    print(f"Total configs: {len(configs)}")

    for cfg, sampled_requests in configs:
        print(f"Current config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        # we scale every time to clean up the task executors states just in case
        # await scale(cfg)
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



