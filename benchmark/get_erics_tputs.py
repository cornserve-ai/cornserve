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
    model_ids = []
    # model_ids.append("OpenGVLab/InternVL3-8B")
    model_ids.append("Qwen/Qwen2.5-VL-7B-Instruct")

    MAX_BATCH_SIZE = 8
    REQUEST_RATE = 15

    app_ids = {}
    for model_id in model_ids:
        group = {}
        for i in range(1, MAX_BATCH_SIZE + 1):
        # for i in range(8,9):
            app_id = ""
            group[i] = app_id
        app_ids[model_id] = group

    configs = []
    image_width = 576
    image_height = 432
    image_count = 8
    num_prompts = 1000

    for model_id in model_ids:
        for bs, app_id in app_ids[model_id].items():
            exp_config = ExperimentConfig(
                backend_config=EricConfig(num_replicas=1, tp_size=1, max_batch_size=bs),
                app_id=app_id,
                model_id=model_id,
                request_rate=REQUEST_RATE,
                # Dedicated Eric profile
                input_len=0,
                output_len=0,
                image_count=image_count,
                num_prompts=num_prompts,
                image_width=image_width,
                image_height=image_height,
                image_choices=32,
            )
            configs.append(exp_config)

    for cfg in configs:
        if cfg.exists():
            print("Batch size", cfg.backend_config.max_batch_size)
            data = cfg.load()
            metrics = data["metrics"]
            completed = metrics["completed"]
            tput = metrics["request_throughput"]
            mean_latency = metrics["mean_e2el_ms"] / 1000
            print("    Completed: {} / {}".format(completed, cfg.num_prompts))
            print("    Throughput: {:.6f} requests/s".format(tput))
            print("    Mean Latency: {:.2f} s".format(mean_latency))


async def main():
    set_ulimit()
    await run(overwrite=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
