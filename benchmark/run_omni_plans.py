"""Run a hard-coded list of Omni experiments.

Each Experiment is constructed directly — no CSV or JSON parsing.

Usage:
    python -u run_omni_plans.py --total-gpus 8
    python -u run_omni_plans.py --total-gpus 8 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Literal, cast

from benchmark_datasets import ServeGenDataset
from benchmark_scheduler import schedule_all
from cornserve_utils import clear_all
from defaults import LLM_GPU_MEMORY_UTILIZATION, OMNI_DURATION
from schema import (
    AppType,
    Experiment,
    OmniApp,
    OmniFlexApp,
    OmniFlexGroupSchema,
    OmniFlexThinkerConfig,
)

# ── Constants ────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SEED = 48105
DEFAULT_GPU_TYPE: Literal["A40", "A100", "H100"] = "A100"

# Workload "mixed" — every request has an image, half include video, half audio.
P_IMG = 1.0
P_VID = 0.5
P_AUDIO = 0.5
P_RETURN_AUDIO = 0.3  # matches plans_omni/8/ra0.3/

# Routing weights for the partial-disagg plan: only request types whose bitmap
# includes the image bit (BIT_IMG=1) route through the image-offloaded group.
IMG_TYPES_TO_GROUP0: dict[int, dict[int, float]] = {
    t: {0: 1.0} for t in (1, 3, 5, 7, 9, 11, 13, 15)
}


# ── Plans ────────────────────────────────────────────────────────────


PLANS: list[tuple[str, AppType, float]] = [
    # (label, app, request_rate)
    (
        "Monolithic: 4× enc_llm (bs=4), 4× tv (bs=64)",
        OmniApp(
            model_id=MODEL_ID,
            encoder_fission=False,
            llm_num_replicas=4,
            llm_tp_size=1,
            llm_max_num_seqs=4,
            llm_gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
            vocoder_fission=False,
            talker_vocoder_num_replicas=4,
            talker_max_num_seqs=64,
        ),
        1,
    ),
    (
        "Full disagg + vocoder fission: 1× img_enc (bs=1), 1× aud_enc (bs=1), 1× vid_enc (bs=1), 1× llm (bs=32), 3× tk (bs=64), 1× geri (bs=1)",
        OmniApp(
            model_id=MODEL_ID,
            encoder_fission=True,
            img_eric_num_replicas=1,
            vid_eric_num_replicas=1,
            audio_eric_num_replicas=1,
            eric_max_batch_size=1,
            llm_num_replicas=1,
            llm_tp_size=1,
            llm_max_num_seqs=32,
            llm_gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
            vocoder_fission=True,
            talker_num_replicas=3,
            audio_geri_num_replicas=1,
            talker_max_num_seqs=64,
            audio_geri_max_batch_size=1,
        ),
        0.8,
    ),
    (
        "Full disagg, fused vocoder: 1× img_enc (bs=1), 1× aud_enc (bs=1), 1× vid_enc (bs=1), 1× llm (bs=64), 4× tv (bs=32)",
        OmniApp(
            model_id=MODEL_ID,
            encoder_fission=True,
            img_eric_num_replicas=1,
            vid_eric_num_replicas=1,
            audio_eric_num_replicas=1,
            eric_max_batch_size=1,
            llm_num_replicas=1,
            llm_tp_size=1,
            llm_max_num_seqs=64,
            llm_gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
            vocoder_fission=False,
            talker_vocoder_num_replicas=4,
            talker_max_num_seqs=32,
        ),
        1,
    ),
    (
        "Monolithic: 2× enc_llm (bs=64), 6× tv (bs=32)",
        OmniApp(
            model_id=MODEL_ID,
            encoder_fission=False,
            llm_num_replicas=2,
            llm_tp_size=1,
            llm_max_num_seqs=64,
            llm_gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
            vocoder_fission=False,
            talker_vocoder_num_replicas=6,
            talker_max_num_seqs=32,
        ),
        1.5,
    ),
    (
        # Partial-disagg / per-type routing — needs OmniFlexApp's per-type routing.
        "Image encoder disagg: 1× img_enc (bs=1), 2× llm (bs=32), 5× tv (bs=64)",
        OmniFlexApp(
            model_id=MODEL_ID,
            groups=[
                OmniFlexGroupSchema(
                    offloaded_modalities=["img"],
                    eric_max_batch_sizes={"img": 1},
                    img_eric_num_replicas=1,
                    vid_eric_num_replicas=0,
                    audio_eric_num_replicas=0,
                    thinkers=[OmniFlexThinkerConfig(
                        llm_num_replicas=2,
                        llm_tp_size=1,
                        llm_max_num_seqs=32,
                        llm_gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
                        weight=1.0,
                    )],
                ),
            ],
            type_routing_weights=IMG_TYPES_TO_GROUP0,
            vocoder_fission=False,
            talker_vocoder_num_replicas=5,
            talker_num_replicas=0,
            audio_geri_num_replicas=0,
            talker_max_num_seqs=64,
        ),
        1.3,
    ),
]


# ── CLI / runner ─────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-gpus", type=int, required=True,
                   help="GPU budget for benchmark scheduling")
    p.add_argument("--gpu-type", choices=["A40", "A100", "H100"],
                   default=DEFAULT_GPU_TYPE)
    p.add_argument("--duration", type=int, default=OMNI_DURATION)
    p.add_argument("--rate-multiplier", type=float, default=1.0,
                   help="Multiplier on the per-plan request rate")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_experiments(
    *, gpu_type: Literal["A40", "A100", "H100"],
    duration: int, rate_multiplier: float, seed: int,
) -> list[Experiment]:
    experiments: list[Experiment] = []
    for _, app, request_rate in PLANS:
        rate = round(rate_multiplier * request_rate, 6)
        dataset = ServeGenDataset(
            request_rate=rate,
            duration=duration,
            image_prob=P_IMG,
            audio_prob=P_AUDIO,
            video_prob=P_VID,
            return_audio_prob=P_RETURN_AUDIO,
            vlm_only=False,
            random_seed=seed,
        )
        experiments.append(
            Experiment(
                app=app,
                dataset=dataset,
                gpu_type=gpu_type,
                seed=seed,
                request_rate=rate,
            )
        )
    return experiments


async def main() -> None:
    args = parse_args()
    gpu_type = cast(Literal["A40", "A100", "H100"], args.gpu_type)

    experiments = build_experiments(
        gpu_type=gpu_type,
        duration=args.duration,
        rate_multiplier=args.rate_multiplier,
        seed=args.seed,
    )

    print(f"Model: {MODEL_ID}  GPU: {gpu_type}  rate_mult: {args.rate_multiplier}")
    for (label, _, request_rate), exp in zip(PLANS, experiments):
        print(f"  request_rate={request_rate:.4f}  rate={exp.request_rate:.4f}  {label}")

    if not args.dry_run:
        await clear_all()

    await schedule_all(
        experiments,
        total_gpus=int(args.total_gpus),
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    asyncio.run(main())
