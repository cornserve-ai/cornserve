from schema import CornserveConfig, EricConfig, ExperimentConfig, VLLMConfig, EPDConfig, PDConfig

import logging
from logging import getLogger

import sys

# stdout logging
logging.basicConfig(
    level="INFO",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = getLogger(__name__)

def run() -> None:

    gpu_type = "A100"
    eric_batch_dize = 1
    vllm_config = VLLMConfig(num_replicas=1, tp_size=2)
    # we compare single vLLM with disaggregated vLLM, ignoring Eric cost
    cornserve_l_config = CornserveConfig(num_vllms=1, vllm_tp_size=2, num_erics=6)
    # isolate Eric
    eric_config = EricConfig(num_replicas=1, tp_size=1, max_batch_size=eric_batch_dize)

    # set max output tokens to 1 to profile prefill 
    epd_p_config = EPDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=4)
    # this might not be optimal
    epd_d_config = EPDConfig(num_prefills=2, prefill_tp_size=2, num_decodes=1, decode_tp_size=2, num_erics=2)

    # set max output tokens to 1 to profile prefill 
    pd_p_config = PDConfig(num_prefills=1, prefill_tp_size=2, num_decodes=3, decode_tp_size=2)

    # the decode in EPD and PD profile can be shared
    pd_d_config = PDConfig(num_prefills=3, prefill_tp_size=2, num_decodes=1, decode_tp_size=2)

    model_id: str = "OpenGVLab/InternVL3-38B"
    # model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    image_width = 1920
    image_height = 1080
    image_count = 1
    input_len = 100
    output_len = 300
    num_prompts = 500
    # model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # image_width = 1920
    # image_height = 1080
    # image_count = 1
    # input_len = 100
    # output_len = 300
    # num_prompts = 500

    configs = []

    for r in [10]:
        eric_exp = ExperimentConfig(
            backend_config=eric_config,
            app_id="",
            model_id=model_id,
            request_rate=r,
            # Dedicated Eric profile
            input_len=0,
            output_len=0,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(eric_exp)

    for r in [5]:
        vllm_exp = ExperimentConfig(
            backend_config=vllm_config,
            app_id="",
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(vllm_exp)

    for r in [5]:
        el_l_exp = ExperimentConfig(
            backend_config=cornserve_l_config,
            app_id="",
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(el_l_exp)


    for r in [5]:
        epd_p_exp = ExperimentConfig(
            backend_config=epd_p_config,
            app_id="",
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            # Dedicated prefill benchmark, so we set output_len to 1
            output_len=1,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(epd_p_exp)

    for r in [5]:
        epd_d_exp = ExperimentConfig(
            backend_config=epd_d_config,
            app_id="",
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(epd_d_exp)

    for r in [5]:
        pd_p_exp = ExperimentConfig(
            backend_config=pd_p_config,
            app_id="",
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            # Dedicated prefill benchmark, so we set output_len to 1
            output_len=1,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(pd_p_exp)

    for r in [5]:
        pd_d_exp = ExperimentConfig(
            backend_config=pd_d_config,
            app_id="",
            model_id=model_id,
            request_rate=r,
            input_len=input_len,
            output_len=output_len,
            image_count=image_count,
            num_prompts=num_prompts,
            image_width=image_width,
            image_height=image_height,
            gpu_type=gpu_type,
        )
        configs.append(pd_d_exp)

    tput_results = {}
    for cfg in configs:
        if cfg.exists():
            print(f"Current config: {cfg.backend_config.__class__.__name__} {cfg.backend_config} {cfg.model_id} with {cfg.request_rate} requests/s")
            data = cfg.load()
            metrics = data["metrics"]
            completed = metrics["completed"]
            total_output = metrics["total_output"]
            tput = metrics["request_throughput"]
            mean_latency = metrics["mean_e2el_ms"] / 1000
            p95_latency = metrics["percentiles_e2el_ms"][1][1] / 1000
            p99_latency = metrics["percentiles_e2el_ms"][2][1] / 1000
            print("    Completed: {} / {}".format(completed, cfg.num_prompts))
            print("    Total output tokens: {}".format(total_output))
            print("    Throughput: {:.2f} requests/s".format(tput))
            print("    Mean Latency: {:.2f} s".format(mean_latency))
            print("    P95 Latency: {:.2f} s".format(p95_latency))
            print("    P99 Latency: {:.2f} s".format(p99_latency))
            tput_results[cfg] = tput

    with open(f"{model_id.replace("/", "_")}_tput_results.json", "w") as f:
        import json
        json.dump({str(k): v for k, v in tput_results.items()}, f, indent=4)

    # we analyze the tputs
    def analyze(gpu_count: int):
        n = gpu_count // 2

        vllm_max_tputs = tput_results[vllm_exp] * n

        # pd
        pd_p_tput = tput_results[pd_p_exp]
        pd_d_tput = tput_results[pd_d_exp]
        pd_tputs = []
        for i in range(1, n):
            pd_tput = min(pd_p_tput * i, pd_d_tput * (n - i))
            logger.debug(f"pd: prefill {i}, decode {n - i}, tput {pd_tput}")
            pd_tputs.append(pd_tput)
        pd_max_tput = max(pd_tputs)

        # epd
        epd_p_tput = tput_results[epd_p_exp]
        epd_d_tput = tput_results[epd_d_exp] if epd_d_exp in tput_results else tput_results[pd_d_exp]
        epd_tputs = []
        for i in range(1, n - 1):
            for j in range(1, n - 1 - i):
                remaining = n - i - j
                eric_tput = tput_results[eric_exp] * remaining * 2
                epd_tput = min(epd_p_tput * i, eric_tput, epd_d_tput * j)
                logger.debug(f"epd: prefill {i}, decode {j}, eric {remaining * 2}, tput {epd_tput}")
                epd_tputs.append(epd_tput)
        epd_max_tput = max(epd_tputs)
        
        # eL
        el_l_tput = tput_results[el_l_exp]
        el_tputs = []
        for i in range(1, n - 1):
            remaining = n - i
            eric_tput = tput_results[eric_exp] * remaining * 2
            el_tput = min(el_l_tput * i, eric_tput)
            logger.debug(f"EL: l {i}, eric {remaining * 2}, tput {el_tput}")
            el_tputs.append(el_tput)
        el_max_tput = max(el_tputs)

        print("For {} GPUs:".format(gpu_count))
        print("    vLLM max throughput: {:.2f} requests/s".format(vllm_max_tputs))
        print("    PD max throughput: {:.2f} requests/s".format(pd_max_tput))
        print("    EPD max throughput: {:.2f} requests/s".format(epd_max_tput))
        print("    EL max throughput: {:.2f} requests/s".format(el_max_tput))

    for gpu_count in range(8, 33, 4):
        analyze(gpu_count)

    cell_size = 128
    L_epd_mappings = {}
    for i in range(2, cell_size + 1, 2):
        # TP2
        num_l = i // 2
        L_epd_mappings[i] = (num_l * tput_results[vllm_exp], (num_l,))

    L_ep_L_d_mappings = {}
    for i in range(2, cell_size + 1, 2):
        num_l = i // 2
        for num_p in range(1, num_l):
            num_d = num_l - num_p
            p_tput = tput_results[pd_p_exp] * num_p
            d_tput = tput_results[pd_d_exp] * num_d
            pd_tput = min(p_tput, d_tput)
            if i not in L_ep_L_d_mappings or pd_tput > L_ep_L_d_mappings[i][0]:
                L_ep_L_d_mappings[i] = (pd_tput, (num_p, num_d))

    E_L_p_L_d_mappings = {}
    for i in range(5, cell_size + 1):
        for num_e in range(1+(i+1)%2, i-4 + 1, 2):
            # start from 1 if odd i, otherwise start from 2
            assert (i - num_e) % 2 == 0
            e_tput = tput_results[eric_exp] * num_e
            num_l = (i - num_e) // 2
            for num_p in range(1, num_l):
                num_d = num_l - num_p
                p_tput = tput_results[epd_p_exp] * num_p
                d_tput = tput_results[pd_d_exp] * num_d
                epd_tput = min(e_tput, p_tput, d_tput)
                if i not in E_L_p_L_d_mappings or epd_tput > E_L_p_L_d_mappings[i][0]:
                    E_L_p_L_d_mappings[i] = (epd_tput, (num_e, num_p, num_d))

    E_L_pd_mappings = {}
    for i in range(3, cell_size + 1):
        for num_e in range(1+(i+1)%2, i-2 + 1, 2):
            assert (i - num_e) % 2 == 0
            e_tput = tput_results[eric_exp] * num_e
            num_l = (i - num_e) // 2
            l_tput = tput_results[el_l_exp] * num_l
            el_tput = min(l_tput, e_tput)
            if i not in E_L_pd_mappings or el_tput > E_L_pd_mappings[i][0]:
                E_L_pd_mappings[i] = (el_tput, (num_e, num_l))
    
    def find(n=32):
        max_tput = 0
        best_cfg = None
        for n_L_epd in range(0, n+1):
            for n_L_ep_L_d in range(0, n - n_L_epd + 1):
                for n_E_L_p_L_d in range(0, n - n_L_epd - n_L_ep_L_d + 1):
                    for n_E_L_pd in range(0, n - n_L_epd - n_L_ep_L_d - n_E_L_p_L_d + 1):
                        tput_n_L_epd, cfg_n_L_epd = L_epd_mappings.get(n_L_epd, (0, (0,)))
                        tput_n_L_ep_L_d, cfg_n_L_ep_L_d = L_ep_L_d_mappings.get(n_L_ep_L_d, (0, (0,0)))
                        tput_n_E_L_p_L_d, cfg_n_E_L_p_L_d = E_L_p_L_d_mappings.get(n_E_L_p_L_d, (0, (0,0,0)))
                        tput_n_E_L_pd, cfg_n_E_L_pd = E_L_pd_mappings.get(n_E_L_pd, (0, (0,0)))
                        total_tput = sum([tput_n_L_epd, tput_n_L_ep_L_d, tput_n_E_L_p_L_d, tput_n_E_L_pd])
                        if total_tput > max_tput:
                            max_tput = total_tput
                            best_cfg = (n_L_epd, cfg_n_L_epd, n_L_ep_L_d, cfg_n_L_ep_L_d, n_E_L_p_L_d, cfg_n_E_L_p_L_d, n_E_L_pd, cfg_n_E_L_pd)
        return max_tput, best_cfg

    for i in range(1,33):
        print(i, find(i))

    # return find




if __name__ == "__main__":
    # find = run()
    run()
