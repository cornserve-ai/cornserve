import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, SyntheticOmniDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EricConfig, ExperimentConfig, OmniConfig, VLLMConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit

async def run(
    overwrite: bool = False,
) -> None:
    # model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_id = "OpenGVLab/InternVL3-38B"

    image_eric = register_app(model_id=model_id, app_type="e", modalities=["IMAGE"])
    print(f"Registered {model_id} image-only Eric with ID: {image_eric}")
    video_eric = register_app(model_id=model_id, app_type="e", modalities=["VIDEO"])
    print(f"Registered {model_id} video-only Eric with ID: {video_eric}")
    el = register_app(model_id=model_id, app_type="ev", modalities=["IMAGE", "VIDEO"])
    print(f"Registered {model_id} omni EV with ID: {el}")

    # vllm = register_app(model_id=model_id, app_type="v")
    # print(f"Registered {model_id} VLLM with ID: {vllm}")
    # vllm_config = VLLMConfig(num_replicas=1)

    image_config = EricConfig(num_replicas=1, modality="image")
    video_config = EricConfig(num_replicas=1, modality="video")
    el_l_config = CornserveConfig(
        num_erics=6,
        num_vllms=1,
        vllm_tp_size=2,
        num_image_erics=2,
        num_video_erics=4,
        modalities=["image", "video"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )

    gpu_type = "A100"
    num_requests = 500
    image_request_rate = 20
    video_request_rate = 20
    llm_request_rate = 2
    input_len = 1000
    # make it short for audio output
    output_len = 100
    image_width = 800
    image_height = 600
    # 609 image tokens
    video_num_frames = 20
    video_width = 1024
    video_height = 768
    # 9990 video tokens

    image_workload_config = OmniConfig(
        include_image=True,
        include_audio=False,
        include_video=False,
        image_width=image_width,
        image_height=image_height,
        video_num_frames=video_num_frames,
        video_height=video_height,
        video_width=video_width,
    )
    video_workload_config = OmniConfig(
        include_image=False,
        include_audio=False,
        include_video=True,
        image_width=image_width,
        image_height=image_height,
        video_num_frames=video_num_frames,
        video_height=video_height,
        video_width=video_width,
    )
    llm_workload_config = OmniConfig(
        include_image=True,
        include_audio=False,
        include_video=True,
        image_width=image_width,
        image_height=image_height,
        video_num_frames=video_num_frames,
        video_height=video_height,
        video_width=video_width,
    )

    def smaple_requests(
        num_requests: int,
        config: OmniConfig,
        input_len: int = 0,
        output_len: int = 0,
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
        )
    image_sampled_requests = smaple_requests(num_requests, image_workload_config)
    video_sampled_requests = smaple_requests(num_requests, video_workload_config)
    llm_sampled_requests = smaple_requests(
        num_requests,
        llm_workload_config,
        input_len=input_len,
        output_len=output_len,
    )

    configs = []
    image_exp_config = ExperimentConfig(
        backend_config=image_config,
        app_id=image_eric,
        model_id=model_id,
        gpu_type=gpu_type,
        dataset="omni",
        workload_config=image_workload_config,
        request_rate=image_request_rate,
        use_synthesized_data=False,
    )
    configs.append((image_exp_config, image_sampled_requests))
    video_exp_config = ExperimentConfig(
        backend_config=video_config,
        app_id=video_eric,
        model_id=model_id,
        gpu_type=gpu_type,
        dataset="omni",
        workload_config=video_workload_config,
        request_rate=video_request_rate,
        use_synthesized_data=False,
    )
    configs.append((video_exp_config, video_sampled_requests))

    el_l_exp_config = ExperimentConfig(
        backend_config=el_l_config,
        app_id=el,
        model_id=model_id,
        gpu_type=gpu_type,
        dataset="omni",
        workload_config=llm_workload_config,
        request_rate=llm_request_rate,
        use_synthesized_data=False,
    )
    configs.append((el_l_exp_config, llm_sampled_requests))

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



