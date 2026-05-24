"""Tests for QwenImage model with Sequence Parallelism (SP).

Follows Eric's testing patterns:
- ``param_sp_size`` decorator for parametrizing SP sizes based on available GPUs
- ``assert_similar()`` for cosine similarity comparison
- pytest fixtures and parametrize decorators

Tests:
- ``test_single_gpu_inference``: Baseline single-GPU generation
- ``test_sp_inference``: SP generation at each supported SP size
- ``test_sp_correctness``: Compare SP output against single-GPU baseline
"""

from __future__ import annotations

import base64
import gc
import io
import multiprocessing as mp
import os
import pickle
import subprocess
from typing import Any

import pytest
import torch
from PIL import Image

from cornserve.task_executors.geri.api import Status
from cornserve.task_executors.geri.models.qwen_image import QwenImageModel

from ..utils import assert_valid_png_results_list, create_dummy_embeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen-Image"
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_STEPS = 4
DEFAULT_SEED = 42
DEFAULT_SEQ_LEN = 77
DEFAULT_BATCH_SIZE = 1

# SP correctness threshold — slightly lower than Eric's 0.98 because
# bfloat16 + All-to-All accumulation across 60 transformer blocks.
SP_COSINE_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# SP size parametrization (mirrors Eric's param_tp_size)
# ---------------------------------------------------------------------------

if (visible_devices := os.getenv("CUDA_VISIBLE_DEVICES")) is not None:
    _CURR_NUM_GPUS = len(visible_devices.split(","))
else:
    try:
        _CURR_NUM_GPUS = int(
            subprocess.check_output(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits", "-i", "0"])
            .strip()
            .decode()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        _CURR_NUM_GPUS = 0

SP_SIZES = [sp for sp in [2, 4, 8] if sp <= _CURR_NUM_GPUS]


def param_sp_size(func):
    """Parametrize test argument ``sp_size`` with power-of-two SP degrees."""
    func = pytest.mark.parametrize(
        "sp_size",
        SP_SIZES,
        ids=lambda x: f"SP={x}",
    )(func)
    return func


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _png_b64_to_tensor(png_b64: str) -> torch.Tensor:
    """Decode a base64 PNG string to a float tensor [C, H, W] in [0, 1]."""
    png_bytes = base64.b64decode(png_b64.encode("ascii"))
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return (
        torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(img.height, img.width, 3).permute(2, 0, 1)
        / 255.0
    )


def _cleanup_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _create_seeded_embeddings(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    seed: int,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Create reproducible dummy prompt embeddings."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return [
        torch.randn(seq_len, hidden_size, dtype=torch.bfloat16, device=device, generator=gen) for _ in range(batch_size)
    ]


def _create_initial_latents(
    pipeline: Any,
    batch_size: int,
    height: int,
    width: int,
    seed: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Create deterministic initial latents for reproducible generation."""
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    return pipeline.prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        torch.bfloat16,
        torch.device(device),
        generator,
    )


# ---------------------------------------------------------------------------
# Subprocess entry points (must be module-level for mp.spawn pickling)
# ---------------------------------------------------------------------------


def _single_gpu_main(result_queue: mp.Queue) -> None:
    """Subprocess: run single-GPU inference via BatchExecutor."""
    from cornserve.task_executors.geri.executor.executor import BatchExecutor

    torch.cuda.set_device(0)
    model = QwenImageModel(
        model_id=MODEL_ID,
        torch_dtype=torch.bfloat16,
        torch_device=torch.device("cuda"),
    )
    executor = BatchExecutor(model=model)

    prompt_embeds = create_dummy_embeddings(batch_size=1)
    result = executor.generate(
        prompt_embeds=prompt_embeds,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        num_inference_steps=DEFAULT_STEPS,
    )

    executor.shutdown()
    result_queue.put(("ok", result.status.value, result.generated))


def _baseline_main(result_queue: mp.Queue) -> None:
    """Subprocess: run baseline generation and return images + initial latents."""
    torch.cuda.set_device(0)

    model = QwenImageModel(
        model_id=MODEL_ID,
        torch_dtype=torch.bfloat16,
        torch_device=torch.device("cuda:0"),
    )

    embeddings = _create_seeded_embeddings(DEFAULT_BATCH_SIZE, DEFAULT_SEQ_LEN, model.embedding_dim, DEFAULT_SEED)

    initial_latents = _create_initial_latents(
        model.pipeline, DEFAULT_BATCH_SIZE, DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_SEED
    )
    initial_latents_cpu = initial_latents.cpu().clone()

    from torch.nn.utils.rnn import pad_sequence

    padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([e.size(0) for e in embeddings], device="cuda:0")
    mask = (torch.arange(padded.size(1), device="cuda:0").unsqueeze(0) < lengths.unsqueeze(1)).long()

    result = model.pipeline(
        prompt_embeds=padded,
        prompt_embeds_mask=mask,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        num_inference_steps=DEFAULT_STEPS,
        latents=initial_latents,
    )

    images_b64 = []
    for img in result.images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    result_queue.put(("ok", images_b64, pickle.dumps(initial_latents_cpu)))


def _sp_worker_main(
    rank: int,
    world_size: int,
    model_id: str,
    height: int,
    width: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    init_method: str,
    result_queue: mp.Queue,
    initial_latents_cpu: torch.Tensor,
) -> None:
    """Subprocess: SP worker entry point for each rank."""
    torch.cuda.set_device(rank)

    from cornserve.task_executors.geri.distributed.parallel import (
        destroy_sp_distributed,
        get_sp_group,
        init_sp_distributed,
    )
    from cornserve.task_executors.geri.models.sp_wrapper import (
        execute_sp_generate,
        load_and_patch_pipeline_for_sp,
    )

    try:
        init_sp_distributed(world_size=world_size, rank=rank, init_method=init_method)
        sp_group = get_sp_group()

        if rank == 0:
            model = QwenImageModel(
                model_id=model_id,
                torch_dtype=torch.bfloat16,
                torch_device=torch.device(f"cuda:{rank}"),
            )
            model.patch_for_sp(sp_group)
            pipeline = model.pipeline
            embedding_dim = model.embedding_dim
        else:
            pipeline, embedding_dim = load_and_patch_pipeline_for_sp(
                model_id=model_id,
                torch_dtype=torch.bfloat16,
                torch_device=torch.device(f"cuda:{rank}"),
                sp_group=sp_group,
            )

        # Create embeddings on rank 0 (same seed as baseline)
        if rank == 0:
            from torch.nn.utils.rnn import pad_sequence

            embeddings = _create_seeded_embeddings(batch_size, seq_len, embedding_dim, seed, device=f"cuda:{rank}")
            prompt_embeds = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
            lengths = torch.tensor([e.size(0) for e in embeddings], device=f"cuda:{rank}")
            prompt_embeds_mask = (
                torch.arange(prompt_embeds.size(1), device=f"cuda:{rank}").unsqueeze(0) < lengths.unsqueeze(1)
            ).long()
        else:
            prompt_embeds = None
            prompt_embeds_mask = None

        result = execute_sp_generate(
            pipeline=pipeline,
            sp_group=sp_group,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            num_inference_steps=steps,
            batch_size=batch_size,
            max_seq_len=seq_len,
            initial_latents=initial_latents_cpu.to(f"cuda:{rank}") if rank == 0 else None,
        )

        if rank == 0:
            result_queue.put(("ok", result))
        else:
            result_queue.put(("ok", None))

    except Exception as e:
        import traceback

        result_queue.put(("error", f"Rank {rank}: {e}\n{traceback.format_exc()}"))
    finally:
        try:
            destroy_sp_distributed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Multi-process helpers
# ---------------------------------------------------------------------------


def _run_sp_generation(
    sp_size: int,
    initial_latents_cpu: torch.Tensor,
    model_id: str = MODEL_ID,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Run SP generation with the given SP size in subprocesses."""
    from cornserve.task_executors.eric.utils.network import get_open_port

    port = get_open_port()
    init_method = f"tcp://127.0.0.1:{port}"

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(sp_size):
        p = ctx.Process(
            target=_sp_worker_main,
            args=(
                rank,
                sp_size,
                model_id,
                height,
                width,
                steps,
                batch_size,
                seq_len,
                seed,
                init_method,
                result_queue,
                initial_latents_cpu,
            ),
        )
        p.start()
        processes.append(p)

    results = []
    for _ in range(sp_size):
        status, data = result_queue.get(timeout=600)
        if status == "error":
            for p in processes:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(f"SP worker failed: {data}")
        results.append(data)

    for p in processes:
        p.join(timeout=30)

    images_b64 = None
    for r in results:
        if r is not None:
            images_b64 = r
            break

    if images_b64 is None:
        raise RuntimeError("No result from rank 0!")

    _cleanup_gpu()
    return images_b64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_results() -> tuple[list[str], torch.Tensor]:
    """Run baseline single-GPU generation in a subprocess.

    The subprocess fully releases GPU memory before returning,
    so SP tests can use all GPUs freely.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    proc = ctx.Process(target=_baseline_main, args=(result_queue,))
    proc.start()

    status, images_b64, latents_bytes = result_queue.get(timeout=300)
    proc.join(timeout=30)

    if status != "ok":
        raise RuntimeError(f"Baseline generation failed: {images_b64}")

    initial_latents_cpu = pickle.loads(latents_bytes)
    return images_b64, initial_latents_cpu


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_gpu_inference() -> None:
    """Baseline single-GPU generation — check Status.SUCCESS via executor.

    Runs in a subprocess to avoid holding GPU memory in the parent pytest process.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    proc = ctx.Process(target=_single_gpu_main, args=(result_queue,))
    proc.start()
    status, result_status, generated = result_queue.get(timeout=300)
    proc.join(timeout=30)

    assert status == "ok"
    assert result_status == Status.SUCCESS.value
    assert_valid_png_results_list(generated, expected_batch_size=1)


@param_sp_size
def test_sp_inference(sp_size: int, baseline_results: tuple[list[str], torch.Tensor]) -> None:
    """SP generation at each supported SP size — check that it produces valid PNGs."""
    _, initial_latents_cpu = baseline_results

    sp_pngs = _run_sp_generation(sp_size=sp_size, initial_latents_cpu=initial_latents_cpu)

    assert_valid_png_results_list(sp_pngs, expected_batch_size=DEFAULT_BATCH_SIZE)


@param_sp_size
def test_sp_correctness(sp_size: int, baseline_results: tuple[list[str], torch.Tensor]) -> None:
    """Compare SP output against single-GPU baseline using cosine similarity.

    Threshold is 0.95 (lower than Eric's 0.98 because bfloat16 + All-to-All
    accumulation across many transformer blocks).
    """
    baseline_pngs, initial_latents_cpu = baseline_results
    baseline_tensors = [_png_b64_to_tensor(b) for b in baseline_pngs]

    sp_pngs = _run_sp_generation(sp_size=sp_size, initial_latents_cpu=initial_latents_cpu)
    sp_tensors = [_png_b64_to_tensor(b) for b in sp_pngs]

    assert len(baseline_tensors) == len(sp_tensors)

    for i in range(len(baseline_tensors)):
        a_flat = baseline_tensors[i].flatten().unsqueeze(0)
        b_flat = sp_tensors[i].flatten().unsqueeze(0)
        cos_sim = torch.cosine_similarity(a_flat, b_flat).item()
        mse = torch.nn.functional.mse_loss(baseline_tensors[i], sp_tensors[i]).item()

        assert cos_sim > SP_COSINE_THRESHOLD, (
            f"SP={sp_size} image {i}: cosine similarity {cos_sim:.6f} < {SP_COSINE_THRESHOLD} (MSE={mse:.8f})"
        )
