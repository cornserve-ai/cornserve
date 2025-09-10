import asyncio

from benchmark_cornserve import benchmark, transform_sampled_requests
from benchmark_dataset import SampleRequest, ServeGenDataset, VisionArenaDataset
from cornserve_utils import register_app, scale
from schema import CornserveConfig, EricConfig, ExperimentConfig, ServeGenConfig, VLLMConfig, EPDConfig, PDConfig
from transformers import AutoTokenizer

from cornserve.utils import set_ulimit


async def run(
    overwrite: bool = False,
) -> None:
    eric_batch_dize = 1
    # model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    model_id: str = "OpenGVLab/InternVL3-38B"
    ev = register_app(model_id=model_id, app_type="ev")
    print(f"Registered {model_id} EV with ID: {ev}")
    e = register_app(model_id=model_id, app_type="e", eric_max_batch_size=eric_batch_dize)
    print(f"Registered {model_id} E with ID: {e}")
    vllm = register_app(model_id=model_id, app_type="v")
    # print(f"Registered {model_id} V with ID: {vllm}")
    # epd = register_app(model_id=model_id, app_type="epd")
    # print(f"Registered {model_id} epd with ID: {epd}")
    # pd = register_app(model_id=model_id, app_type="pd")
    # print(f"Registered {model_id} pd with ID: {pd}")

    vllm_config = VLLMConfig(num_replicas=1, tp_size=2)
    # we compare single vLLM with disaggregated vLLM, ignoring Eric cost
    cornserve_l_config = CornserveConfig(num_vllms=1, vllm_tp_size=2, num_erics=6)

    # isolate Eric
    eric_config = EricConfig(num_replicas=1, tp_size=1, max_batch_size=eric_batch_dize)

    # # set max output tokens to 1 to profile prefill 
    # epd_p_config = EPDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=4)
    # # this might not be optimal
    # epd_d_config = EPDConfig(num_prefills=2, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=2)
    #
    # # set max output tokens to 1 to profile prefill 
    # pd_p_config = PDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=3, decode_tp_size=2)
    # pd_d_config = PDConfig(num_prefills=3, prefill_tp_size=2, num_decodes=1, decode_tp_size=2)

    configs = []
    gpu_type = "A100"
    # request rate -> sampled_requests
    sampled_workloads = {}
    # we use a large request rate
    duration = 300
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=True,
    )

    def _try_sample_workload(r: int) -> list[SampleRequest]:
        if r in sampled_workloads:
            return sampled_workloads[r]
        serve_gen_config = ServeGenConfig(
            request_rate=r,
            duration=duration,
        )
        print(f"Sampling reqeuests ...")
        workload: list[SampleRequest] = ServeGenDataset().sample(
            tokenizer=tokenizer,
            request_rate=serve_gen_config.request_rate,
            duration=serve_gen_config.duration,
            no_image_prob=serve_gen_config.no_image_prob,
            audio_prob=serve_gen_config.audio_prob,
            video_prob=serve_gen_config.video_prob,
        )
        sampled_workloads[r] = workload
        return workload


    for r in [10]:
        serve_gen_config = ServeGenConfig(request_rate=r, duration=duration)
        sampled_workloads[r] = _try_sample_workload(r)
        exp_config = ExperimentConfig(
            backend_config=eric_config,
            app_id=e,
            model_id=model_id,
            request_rate=r,
            gpu_type=gpu_type,
            dataset="servegen",
            # Dedicated Eric profile
            input_len=0,
            output_len=0,
            # serve gen
            workload_config=serve_gen_config,
            use_synthesized_data=False,
        )
        configs.append(exp_config)

    # then we run L_{PD} bc CUDA IPC somehow always use GPU 0 (bug?) and having high GPU utilization
    # may cause CUDA OOM
    for r in [5]:
        serve_gen_config = ServeGenConfig(request_rate=r, duration=duration)
        sampled_workloads[r] = _try_sample_workload(r)
        exp_config = ExperimentConfig(
            backend_config=cornserve_l_config,
            app_id=ev,
            model_id=model_id,
            request_rate=r,
            gpu_type=gpu_type,
            dataset="servegen",
            # serve gen
            workload_config=serve_gen_config,
            use_synthesized_data=False,
        )
        configs.append(exp_config)

    for r in [5]:
        serve_gen_config = ServeGenConfig(request_rate=r, duration=duration)
        sampled_workloads[r] = _try_sample_workload(r)
        exp_config = ExperimentConfig(
            backend_config=vllm_config,
            app_id=vllm,
            model_id=model_id,
            request_rate=r,
            gpu_type=gpu_type,
            dataset="servegen",
            # serve gen
            workload_config=serve_gen_config,
            use_synthesized_data=False,
        )
        configs.append(exp_config)

    if not overwrite:
        configs = [cfg for cfg in configs if not cfg.exists()]

    # prioritize by request rate
    configs.sort(key=lambda config: (-config.request_rate,))

    print(f"Total configs: {len(configs)}")

    for cfg in configs:
        print(f"Current config: {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
        serve_gen_rr = cfg.workload_config.request_rate
        sampled_requests = sampled_workloads[serve_gen_rr]
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


async def main():
    set_ulimit()
    await run(overwrite=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")

