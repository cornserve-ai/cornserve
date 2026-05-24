"""Unified benchmark runner for Cornserve apps (Eric, MLLM, Geri).

Handles three app types through a single ``benchmark()`` entry point:
- **Eric** (DummyEricApp): non-streaming encoder profiling
- **MLLM** (DummyLLMApp / MonolithicLLMApp / router/composite MLLM apps): streaming LLM with TTFT/ITL
- **Geri** (DummyImageGeriApp / DummyAudioGeriApp): generator profiling

All apps share: Poisson/timestamp dispatching, OTel trace collection,
queuing-adjusted latency stats, and Experiment-based result persistence.
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
import numpy as np
from tqdm import tqdm

from benchmark_datasets import ControlledDiT, OmniRequest, VLMRequest
from cornserve_utils import GATEWAY_URL, ExecutorLogStreamer
from prometheus_utils import (
    PortForwardManager,
    PrometheusCollector,
    calculate_steady_state_stats,
    executor_pod_prefixes,
    prometheus_pod_prefixes,
    save_timeline,
)
from request_tracker import RequestTracker
from otel_utils import (
    DisaggImageOtelProcessor,
    EPDMLLMOtelProcessor,
    EricOtelProcessor,
    GeriOtelProcessor,
    LLMOtelProcessor,
    MLLMOtelProcessor,
    OmniRouterOtelProcessor,
    OtelProcessor,
    PDMLLMOtelProcessor,
    ProcessingMetrics,
    TalkerOtelProcessor,
    TalkerVocoderOtelProcessor,
)
from schema import (
    DecodeLLMApp,
    DummyAudioGeriApp,
    DummyEricApp,
    DummyImageGeriApp,
    DummyLLMApp,
    DummyTalkerApp,
    DummyTalkerVocoderApp,
    EPDMLLMApp,
    Experiment,
    GroupedMixedMLLMApp,
    MixedMLLMApp,
    MixedMonoQwenImageApp,
    MixedQwenImageDisaggApp,
    MLLMRouterApp,
    MLLMRouterMixedRouteConfig,
    ModServeApp,
    MonolithicLLMApp,
    OmniMLLMApp,
    OmniFlexApp,
    OmniRouterApp,
    QwenImageDisaggApp,
    QwenImageTextGeriApp,
    SuperGroupQwenImageApp,
    PDMLLMApp,
    PrefillLLMApp,
    TimeSharingApp,
)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
AUDIO_TOKEN_EXPANSION_FACTOR = 4

# Raise the open-file limit to avoid "Too many open files" when hundreds of
# streaming connections are in flight concurrently.
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except (ValueError, OSError):
    # Fall back to hard limit if unlimited is not permitted.
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _soft < _hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EricInput:
    """Input for a single Eric benchmark request."""

    url: str
    model_id: str
    data_urls: list[str]
    timestamp: float | None = None


@dataclass
class MLLMInput:
    """Input for a single MLLM benchmark request (OpenAI chat format)."""

    url: str
    model_id: str
    content: list[dict[str, Any]]
    output_len: int
    prompt_len: int = 0
    timestamp: float | None = None
    image_resolutions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class OmniMLLMInput:
    """Input for a single Omni MLLM benchmark request (with return_audio)."""

    url: str
    model_id: str
    content: list[dict[str, Any]]
    output_len: int
    return_audio: bool = False
    prompt_len: int = 0
    timestamp: float | None = None
    image_resolutions: list[tuple[int, int]] = field(default_factory=list)
    video_metadata: list[dict[str, int]] = field(default_factory=list)
    audio_durations_s: list[int] = field(default_factory=list)


@dataclass
class GeriInput:
    """Input for a single Geri benchmark request."""

    url: str
    request_data: dict[str, Any]
    timestamp: float | None = None


@dataclass
class TalkerInput:
    """Input for a single Omni Talker embedding benchmark request."""

    url: str
    model_id: str
    prompt: str
    thinker_hidden_states_len: int
    thinker_output_len: int
    max_completion_tokens: int
    timestamp: float | None = None


@dataclass
class RequestResult:
    """Result of a single benchmark request.

    Base fields are populated by all app types.  MLLM-specific fields
    (ttft, itl, output_tokens, etc.) default to zero/empty for non-MLLM apps.
    """

    success: bool = False
    latency: float = 0.0
    error: str = ""
    trace_id: str = ""
    queuing_delay: float = 0.0  # server-side queuing (s), filled from OTel
    send_time: float = 0.0  # perf_counter offset from benchmark_start
    # MLLM-specific
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    output_tokens: int = 0
    prompt_len: int = 0
    generated_text: str = ""
    image_resolutions: list[tuple[int, int]] = field(default_factory=list)
    # Omni-specific: per-request modality info
    return_audio: bool = False
    num_images: int = 0
    num_videos: int = 0
    num_audios: int = 0
    video_metadata: list[dict[str, int]] = field(default_factory=list)  # [{frames, height, width}]
    audio_durations_s: list[int] = field(default_factory=list)  # seconds per audio


# ---------------------------------------------------------------------------
# Input transforms
# ---------------------------------------------------------------------------


def to_eric_inputs(
    requests: list[VLMRequest] | list[OmniRequest],
    app_id: str,
    model_id: str,
) -> list[EricInput]:
    """Transform dataset requests into EricInputs for Eric invocation."""
    url = f"{GATEWAY_URL}/app/invoke/{app_id}"
    inputs: list[EricInput] = []
    for req in requests:
        data_urls: list[str] = []
        for mm_entry in req.multi_modal_data_list:
            mm_type = mm_entry["type"].split("_")[0]
            uri_key = f"{mm_type}_url"
            if uri_key in mm_entry:
                data_urls.append(mm_entry[uri_key]["url"])
        inputs.append(
            EricInput(
                url=url, model_id=model_id, data_urls=data_urls, timestamp=req.timestamp
            )
        )
    return inputs


def _parse_image_resolutions(
    filenames: list[tuple[str, str]],
) -> list[tuple[int, int]]:
    """Extract (width, height) from image filenames like '1080x1920_0.png'."""
    resolutions = []
    for modality, fname in filenames:
        if modality == "image":
            # fname format: "{min_dim}x{max_dim}_{id}.png"
            stem = fname.rsplit("_", 1)[0]  # e.g. "1080x1920"
            parts = stem.split("x")
            h, w = int(parts[0]), int(parts[1])
            resolutions.append((w, h))
    return resolutions


def _parse_video_metadata(
    filenames: list[tuple[str, str]],
) -> list[dict[str, int]]:
    """Extract {frames, height, width} from video filenames like '8x1036x1736_0.mp4'."""
    metadata = []
    for modality, fname in filenames:
        if modality == "video":
            # fname format: "{frames}x{height}x{width}_{id}.mp4"
            stem = fname.rsplit("_", 1)[0]
            parts = stem.split("x")
            metadata.append({"frames": int(parts[0]), "height": int(parts[1]), "width": int(parts[2])})
    return metadata


def _parse_audio_durations(
    filenames: list[tuple[str, str]],
) -> list[int]:
    """Extract duration in seconds from audio filenames like '7s_0.wav'."""
    durations = []
    for modality, fname in filenames:
        if modality == "audio":
            # fname format: "{seconds}s_{id}.wav"
            stem = fname.rsplit("_", 1)[0]  # e.g. "7s"
            durations.append(int(stem.rstrip("s")))
    return durations


def to_mllm_inputs(
    requests: list[VLMRequest],
    app_id: str,
    model_id: str,
) -> list[MLLMInput]:
    """Transform VLMRequests into MLLMInputs (OpenAI chat format)."""
    url = f"{GATEWAY_URL}/app/invoke/{app_id}"
    inputs: list[MLLMInput] = []
    for req in requests:
        content: list[dict[str, Any]] = [{"type": "text", "text": req.prompt}]
        content.extend(req.multi_modal_data_list)
        inputs.append(
            MLLMInput(
                url=url,
                model_id=model_id,
                content=content,
                output_len=req.output_len,
                prompt_len=req.prompt_len,
                timestamp=req.timestamp,
                image_resolutions=_parse_image_resolutions(req.filenames),
            )
        )
    return inputs


def to_omni_inputs(
    requests: list[OmniRequest],
    app_id: str,
    model_id: str,
) -> list[OmniMLLMInput]:
    """Transform OmniRequests into OmniMLLMInputs (with return_audio)."""
    url = f"{GATEWAY_URL}/app/invoke/{app_id}"
    inputs: list[OmniMLLMInput] = []
    for req in requests:
        content: list[dict[str, Any]] = [{"type": "text", "text": req.prompt}]
        content.extend(req.multi_modal_data_list)
        inputs.append(
            OmniMLLMInput(
                url=url,
                model_id=model_id,
                content=content,
                output_len=req.output_len,
                return_audio=req.return_audio,
                prompt_len=req.prompt_len,
                timestamp=req.timestamp,
                image_resolutions=_parse_image_resolutions(req.filenames),
                video_metadata=_parse_video_metadata(req.filenames),
                audio_durations_s=_parse_audio_durations(req.filenames),
            )
        )
    return inputs


def to_geri_audio_inputs(
    requests: list[OmniRequest],
    app_id: str,
) -> list[GeriInput]:
    """Transform OmniRequests into GeriInputs for audio Geri.

    We expand the dummy sequence length relative to ServeGen output length to
    match the expected Talker->Vocoder token inflation in audio generation.
    """
    url = f"{GATEWAY_URL}/app/invoke/{app_id}"
    return [
        GeriInput(
            url=url,
            request_data={
                "dummy_seq_len": req.output_len * AUDIO_TOKEN_EXPANSION_FACTOR
            },
            timestamp=req.timestamp,
        )
        for req in requests
    ]


def to_talker_inputs(
    requests: list[OmniRequest],
    app_id: str,
    model_id: str,
) -> list[TalkerInput]:
    """Transform OmniRequests into TalkerInputs for the Omni Talker embedding task.

    For qwen3 omni profiling:
    - thinker_hidden_states_len = ServeGen input + output lengths
    - thinker_output_len = ServeGen output length
    - max_completion_tokens = 4x ServeGen output length
    """
    url = f"{GATEWAY_URL}/app/invoke/{app_id}"
    return [
        TalkerInput(
            url=url,
            model_id=model_id,
            prompt=req.prompt if isinstance(req.prompt, str) else "",
            thinker_hidden_states_len=req.prompt_len + req.output_len,
            thinker_output_len=req.output_len,
            max_completion_tokens=req.output_len * AUDIO_TOKEN_EXPANSION_FACTOR,
            timestamp=req.timestamp,
        )
        for req in requests
    ]


# ---------------------------------------------------------------------------
# Traceparent helper
# ---------------------------------------------------------------------------


def _generate_traceparent() -> tuple[str, str]:
    """Generate a W3C traceparent header and return (trace_id, header)."""
    trace_id = os.urandom(16).hex()
    span_id = os.urandom(8).hex()
    return trace_id, f"00-{trace_id}-{span_id}-01"


# ---------------------------------------------------------------------------
# Invoke functions
# ---------------------------------------------------------------------------


async def _invoke_app(
    url: str,
    payload: dict[str, Any],
    session: aiohttp.ClientSession | None = None,
) -> RequestResult:
    """POST *payload* to *url* with traceparent, read full response.

    Used by Eric and Geri (non-streaming).
    """
    trace_id, traceparent = _generate_traceparent()
    result = RequestResult(trace_id=trace_id)
    st = time.perf_counter()
    try:
        if session is None:
            async with aiohttp.ClientSession(
                trust_env=True, timeout=AIOHTTP_TIMEOUT
            ) as sess:
                async with sess.post(
                    url=url,
                    json=payload,
                    headers={"traceparent": traceparent},
                ) as response:
                    if response.status == 200:
                        await response.read()
                        result.success = True
                    else:
                        body = await response.text()
                        result.error = f"HTTP {response.status}: {body[:500]}"
        else:
            async with session.post(
                url=url,
                json=payload,
                headers={"traceparent": traceparent},
            ) as response:
                if response.status == 200:
                    await response.read()
                    result.success = True
                else:
                    body = await response.text()
                    result.error = f"HTTP {response.status}: {body[:500]}"
    except Exception:
        result.error = "".join(traceback.format_exception(*sys.exc_info()))
    result.latency = time.perf_counter() - st
    return result


async def invoke_eric(
    inp: EricInput,
    session: aiohttp.ClientSession | None = None,
    tracker: RequestTracker | None = None,  # noqa: ARG001
) -> RequestResult:
    """Send a single request to an Eric app."""
    payload = {"request_data": {"model_id": inp.model_id, "data_urls": inp.data_urls}}
    return await _invoke_app(inp.url, payload, session)


async def invoke_geri(
    inp: GeriInput,
    session: aiohttp.ClientSession | None = None,
    tracker: RequestTracker | None = None,  # noqa: ARG001
) -> RequestResult:
    """Send a single request to a Geri app."""
    payload = {"request_data": inp.request_data}
    return await _invoke_app(inp.url, payload, session)


async def invoke_talker(
    inp: TalkerInput,
    session: aiohttp.ClientSession | None = None,
    tracker: RequestTracker | None = None,  # noqa: ARG001
) -> RequestResult:
    """Send a single non-streaming request to an Omni Talker embedding app."""
    payload = {
        "request_data": {
            "model": inp.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": inp.prompt or "Hello"}],
                },
                {"role": "assistant", "content": "<tts_pad>"},
            ],
            "thinker_hidden_states_len": inp.thinker_hidden_states_len,
            "thinker_output_len": inp.thinker_output_len,
            "max_completion_tokens": inp.max_completion_tokens,
            "ignore_eos": True,
        }
    }
    return await _invoke_app(inp.url, payload, session)


async def invoke_talker_vocoder(
    inp: TalkerInput,
    session: aiohttp.ClientSession | None = None,
    tracker: RequestTracker | None = None,
) -> RequestResult:
    """Send a streaming request to an Omni Talker Vocoder app.

    Streams audio chunks as SSE events. Measures time-to-first-chunk (TTFT)
    and inter-chunk latency (ITL), matching the invoke_mllm pattern.
    """
    payload = {
        "request_data": {
            "model": inp.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": inp.prompt or "Hello"}],
                },
                {"role": "assistant", "content": "<tts_pad>"},
            ],
            "thinker_hidden_states_len": inp.thinker_hidden_states_len,
            "thinker_output_len": inp.thinker_output_len,
            "max_completion_tokens": inp.max_completion_tokens,
            "ignore_eos": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
    }

    trace_id, traceparent = _generate_traceparent()
    result = RequestResult(trace_id=trace_id)

    ttft = 0.0
    current_completion_tokens = 0
    st = time.perf_counter()
    most_recent_timestamp = st

    async def _do_request(sess: aiohttp.ClientSession) -> None:
        nonlocal ttft, current_completion_tokens, most_recent_timestamp
        async with sess.post(
            url=inp.url,
            json=payload,
            headers={"traceparent": traceparent},
        ) as response:
            if response.status == 200:
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        timestamp = time.perf_counter()

                        usage = data.get("usage")
                        completion_tokens = (
                            usage.get("completion_tokens") if usage else None
                        )

                        if data.get("choices"):
                            if ttft == 0.0:
                                ttft = timestamp - st
                                result.ttft = ttft
                                if tracker is not None:
                                    tracker.notify_first_token()
                            else:
                                result.itl.append(timestamp - most_recent_timestamp)

                            if completion_tokens is not None:
                                inc = completion_tokens - current_completion_tokens
                                current_completion_tokens = completion_tokens
                                if tracker is not None and inc > 0:
                                    tracker.notify_tokens_generated(inc)
                            elif tracker is not None:
                                tracker.notify_tokens_generated(1)

                        elif usage:
                            result.output_tokens = usage.get("completion_tokens", 0)

                        most_recent_timestamp = timestamp

                result.success = True
                result.latency = most_recent_timestamp - st
            else:
                body = await response.text()
                result.error = f"HTTP {response.status}: {body[:500]}"

    try:
        if session is not None:
            await _do_request(session)
        else:
            connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(
                trust_env=True, timeout=AIOHTTP_TIMEOUT, connector=connector
            ) as fallback_session:
                await _do_request(fallback_session)
    except Exception:
        result.error = "".join(traceback.format_exception(*sys.exc_info()))
        result.latency = time.perf_counter() - st
    finally:
        if tracker is not None:
            tracker.notify_request_finished(result.success)

    return result


async def invoke_mllm(
    inp: MLLMInput,
    session: aiohttp.ClientSession | None = None,
    tracker: RequestTracker | None = None,
) -> RequestResult:
    """Send a streaming request to an MLLM app.

    Parses SSE events, measures TTFT and per-token ITL.  When a single
    SSE chunk carries multiple tokens (via continuous_usage_stats),
    one real ITL is recorded followed by (N-1) zeros.

    If *tracker* is provided, it is notified on first-token, token
    generation, and request completion for steady-state detection.
    """
    payload = {
        "request_data": {
            "model": inp.model_id,
            "messages": [{"role": "user", "content": inp.content}],
            "max_completion_tokens": inp.output_len,
            "ignore_eos": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
    }

    trace_id, traceparent = _generate_traceparent()
    result = RequestResult(
        trace_id=trace_id,
        prompt_len=inp.prompt_len,
        image_resolutions=inp.image_resolutions,
    )

    generated_text = ""
    ttft = 0.0
    current_completion_tokens = 0
    st = time.perf_counter()
    most_recent_timestamp = st

    async def _do_request(sess: aiohttp.ClientSession) -> None:
        nonlocal generated_text, ttft, current_completion_tokens, most_recent_timestamp
        async with sess.post(
            url=inp.url,
            json=payload,
            headers={"traceparent": traceparent},
        ) as response:
            if response.status == 200:
                buffer = b""
                done = False
                async for chunk in response.content.iter_any():
                    if done:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8").strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        if not line:
                            continue
                        if line == "[DONE]":
                            done = True
                            break

                        data = json.loads(line)
                        timestamp = time.perf_counter()

                        usage = data.get("usage")
                        completion_tokens = (
                            usage.get("completion_tokens") if usage else None
                        )

                        if choices := data.get("choices"):
                            content_text = choices[0].get("delta", {}).get("content")
                            if ttft == 0.0:
                                ttft = timestamp - st
                                result.ttft = ttft
                                if tracker is not None:
                                    tracker.notify_first_token()
                            else:
                                result.itl.append(timestamp - most_recent_timestamp)

                            if completion_tokens is not None:
                                inc = completion_tokens - current_completion_tokens
                                current_completion_tokens = completion_tokens
                                if tracker is not None and inc > 0:
                                    tracker.notify_tokens_generated(inc)
                                for _ in range(inc - 1):
                                    result.itl.append(0.0)
                            elif tracker is not None:
                                # No continuous_usage_stats — count 1 token per chunk
                                tracker.notify_tokens_generated(1)

                            generated_text += content_text or ""

                        elif usage:
                            result.output_tokens = usage.get("completion_tokens", 0)

                        most_recent_timestamp = timestamp

                result.generated_text = generated_text
                result.success = True
                result.latency = most_recent_timestamp - st
            else:
                body = await response.text()
                result.error = f"HTTP {response.status}: {body[:500]}"

    try:
        if session is not None:
            await _do_request(session)
        else:
            connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(
                trust_env=True, timeout=AIOHTTP_TIMEOUT, connector=connector
            ) as fallback_session:
                await _do_request(fallback_session)
    except Exception:
        result.error = "".join(traceback.format_exception(*sys.exc_info()))
        result.latency = time.perf_counter() - st
    finally:
        if tracker is not None:
            tracker.notify_request_finished(result.success)

    return result


async def invoke_omni(
    inp: OmniMLLMInput,
    session: aiohttp.ClientSession | None = None,
    tracker: RequestTracker | None = None,
) -> RequestResult:
    """Send a streaming request to an Omni MLLM app (with return_audio).

    Same as invoke_mllm but adds return_audio to the request payload.
    """
    payload = {
        "request_data": {
            "model": inp.model_id,
            "messages": [{"role": "user", "content": inp.content}],
            "max_completion_tokens": inp.output_len,
            "ignore_eos": True,
            "return_audio": inp.return_audio,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        }
    }

    # Count modalities from content
    num_images = sum(1 for c in inp.content if c.get("type") == "image_url")
    num_videos = sum(1 for c in inp.content if c.get("type") == "video_url")
    num_audios = sum(1 for c in inp.content if c.get("type") == "audio_url")

    trace_id, traceparent = _generate_traceparent()
    result = RequestResult(
        trace_id=trace_id,
        prompt_len=inp.prompt_len,
        image_resolutions=inp.image_resolutions,
        return_audio=inp.return_audio,
        num_images=num_images,
        num_videos=num_videos,
        num_audios=num_audios,
        video_metadata=inp.video_metadata,
        audio_durations_s=inp.audio_durations_s,
    )

    generated_text = ""
    ttft = 0.0
    current_completion_tokens = 0
    st = time.perf_counter()
    most_recent_timestamp = st

    async def _do_request(sess: aiohttp.ClientSession) -> None:
        nonlocal generated_text, ttft, current_completion_tokens, most_recent_timestamp
        async with sess.post(
            url=inp.url,
            json=payload,
            headers={"traceparent": traceparent},
        ) as response:
            if response.status == 200:
                buffer = b""
                done = False
                async for chunk in response.content.iter_any():
                    if done:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode("utf-8").strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        if not line:
                            continue
                        if line == "[DONE]":
                            done = True
                            break

                        data = json.loads(line)
                        timestamp = time.perf_counter()

                        usage = data.get("usage")
                        completion_tokens = (
                            usage.get("completion_tokens") if usage else None
                        )

                        if choices := data.get("choices"):
                            content_text = choices[0].get("delta", {}).get("content")
                            if ttft == 0.0:
                                ttft = timestamp - st
                                result.ttft = ttft
                                if tracker is not None:
                                    tracker.notify_first_token()
                            else:
                                result.itl.append(timestamp - most_recent_timestamp)

                            if completion_tokens is not None:
                                inc = completion_tokens - current_completion_tokens
                                current_completion_tokens = completion_tokens
                                if tracker is not None and inc > 0:
                                    tracker.notify_tokens_generated(inc)
                                for _ in range(inc - 1):
                                    result.itl.append(0.0)
                            elif tracker is not None:
                                tracker.notify_tokens_generated(1)

                            generated_text += content_text or ""

                        elif usage:
                            result.output_tokens = usage.get("completion_tokens", 0)

                        most_recent_timestamp = timestamp

                result.generated_text = generated_text
                result.success = True
                result.latency = most_recent_timestamp - st
            else:
                body = await response.text()
                result.error = f"HTTP {response.status}: {body[:500]}"

    try:
        if session is not None:
            await _do_request(session)
        else:
            connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(
                trust_env=True, timeout=AIOHTTP_TIMEOUT, connector=connector
            ) as fallback_session:
                await _do_request(fallback_session)
    except Exception:
        result.error = "".join(traceback.format_exception(*sys.exc_info()))
        result.latency = time.perf_counter() - st
    finally:
        if tracker is not None:
            tracker.notify_request_finished(result.success)

    return result


# ---------------------------------------------------------------------------
# Request dispatching
# ---------------------------------------------------------------------------


async def dispatch_requests(
    inputs: list[Any],
    invoke_fn: Callable[..., Awaitable[RequestResult]],
    request_rate: float = float("inf"),
    max_concurrency: int | None = None,
    rng: np.random.Generator | None = None,
    tracker: RequestTracker | None = None,
    benchmark_start: float | None = None,
    cancel_event: asyncio.Event | None = None,
) -> list[RequestResult]:
    """Dispatch *inputs* through *invoke_fn* with controlled arrival timing.

    - If requests carry ``timestamp`` values, each request sleeps until its
      timestamp (relative to the start of dispatching) before being sent.
    - Otherwise, inter-arrival times follow a Poisson process at
      *request_rate* req/s.
    - *max_concurrency*, when set, limits how many requests are in-flight
      at once via an ``asyncio.Semaphore``.
    - *rng*, when provided, is a per-experiment ``np.random.Generator``
      used for Poisson inter-arrival sampling so that concurrent
      experiments don't race on global numpy state.
    - *tracker*, when provided, is passed to *invoke_fn* (as keyword arg
      ``tracker``) for steady-state detection in MLLM benchmarks.
    - *benchmark_start*, when provided, is the ``perf_counter`` reference
      for computing ``send_time`` on each result.  If ``None``, captured
      at the start of this function.
    - *cancel_event*, when provided and set, aborts the benchmark early:
      stops dispatching new requests and cancels all in-flight tasks.
    """
    if rng is None:
        rng = np.random.default_rng()
    use_timestamps = (
        getattr(inputs[0], "timestamp", None) is not None if inputs else False
    )
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    pbar = tqdm(total=len(inputs), desc="Dispatching requests")
    t0 = benchmark_start if benchmark_start is not None else time.perf_counter()

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT, connector=connector
    ) as session:

        async def _guarded(inp: Any, send_time: float) -> RequestResult:
            if semaphore is not None:
                async with semaphore:
                    res = await invoke_fn(inp, session, tracker=tracker)
            else:
                res = await invoke_fn(inp, session, tracker=tracker)
            res.send_time = send_time
            pbar.update(1)
            return res

        tasks: list[asyncio.Task[RequestResult]] = []
        cancelled = False

        if use_timestamps:

            async def _timed(inp: Any) -> RequestResult:
                await asyncio.sleep(inp.timestamp)
                return await _guarded(inp, time.perf_counter() - t0)

            for inp in inputs:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                tasks.append(asyncio.create_task(_timed(inp)))
        else:
            for inp in inputs:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                send_time = time.perf_counter() - t0
                tasks.append(asyncio.create_task(_guarded(inp, send_time)))
                if request_rate != float("inf"):
                    interval = rng.exponential(1.0 / request_rate)
                    await asyncio.sleep(interval)

        if cancelled:
            print(
                f"\n*** Cancelling {len(tasks)} in-flight requests "
                f"due to pod failure ***"
            )
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()
            raise RuntimeError("Benchmark aborted: executor pod died during dispatch")

        # If a cancel_event is provided, watch for it during gather so we
        # can abort even after all requests have been dispatched.
        if cancel_event is not None:

            async def _cancel_watcher() -> None:
                await cancel_event.wait()
                print(
                    f"\n*** Pod failure detected — cancelling "
                    f"{sum(1 for t in tasks if not t.done())} in-flight requests ***"
                )
                for t in tasks:
                    t.cancel()

            watcher = asyncio.create_task(_cancel_watcher())
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass

            if cancel_event.is_set():
                pbar.close()
                raise RuntimeError(
                    "Benchmark aborted: executor pod died during dispatch"
                )
            # Filter out should-not-happen exceptions (safety net).
            final: list[RequestResult] = []
            for r in results:
                if isinstance(r, BaseException):
                    raise r
                final.append(r)
            pbar.close()
            return final

        results = await asyncio.gather(*tasks)
    pbar.close()
    return list(results)


# ---------------------------------------------------------------------------
# OTel collection
# ---------------------------------------------------------------------------


def _make_otel_processor(app: Any, app_id: str) -> OtelProcessor:
    """Create the appropriate OtelProcessor for *app*."""
    if isinstance(app, DummyEricApp):
        return EricOtelProcessor(app_id=app_id, modality=app.modality)
    elif isinstance(app, (OmniRouterApp, OmniFlexApp)):
        return OmniRouterOtelProcessor(app_id=app_id)
    elif isinstance(
        app,
        (ModServeApp, TimeSharingApp, MLLMRouterApp, MixedMLLMApp, GroupedMixedMLLMApp),
    ):
        return MLLMOtelProcessor(app_id=app_id)
    elif isinstance(app, EPDMLLMApp):
        return EPDMLLMOtelProcessor(app_id=app_id)
    elif isinstance(app, PDMLLMApp):
        return PDMLLMOtelProcessor(app_id=app_id)
    elif isinstance(app, (DummyLLMApp, MonolithicLLMApp, OmniMLLMApp, PrefillLLMApp, DecodeLLMApp)):
        return LLMOtelProcessor(app_id=app_id)
    elif isinstance(app, DummyImageGeriApp):
        return GeriOtelProcessor(app_id=app_id, modality="image")
    elif isinstance(app, (QwenImageDisaggApp, MixedQwenImageDisaggApp,
                          SuperGroupQwenImageApp)):
        return DisaggImageOtelProcessor(app_id=app_id)
    elif isinstance(app, (QwenImageTextGeriApp, MixedMonoQwenImageApp)):
        return GeriOtelProcessor(app_id=app_id, modality="image")
    elif isinstance(app, DummyAudioGeriApp):
        return GeriOtelProcessor(app_id=app_id, modality="audio")
    elif isinstance(app, DummyTalkerApp):
        return TalkerOtelProcessor(app_id=app_id)
    elif isinstance(app, DummyTalkerVocoderApp):
        return TalkerVocoderOtelProcessor(app_id=app_id)
    else:
        raise ValueError(f"No OTel processor for {type(app).__name__}")


def _process_traces_sync(
    otel_processor: OtelProcessor,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Process OTel traces (CPU-bound, run in a thread)."""
    trace_metrics_by_id: dict[str, Any] = {}
    for trace in traces:
        for tm in otel_processor._process_trace(trace):
            trace_metrics_by_id[tm.trace_id] = tm
    return trace_metrics_by_id


def _save_and_summarize_traces_sync(
    otel_processor: OtelProcessor,
    traces: list[dict[str, Any]],
    otel_output_dir: Path,
) -> ProcessingMetrics:
    """Save raw traces and compute summary (IO+CPU-bound, run in a thread)."""
    otel_output_dir.mkdir(parents=True, exist_ok=True)
    # Save all raw traces as a single Jaeger-compatible JSON file
    with open(otel_output_dir / "raw.json", "w") as f:
        json.dump({"data": traces}, f, indent=2)
    return otel_processor._summarize_and_save(traces, otel_output_dir)


async def _collect_otel(
    results: list[RequestResult],
    otel_processor: OtelProcessor,
    otel_output_dir: Path,
) -> tuple[ProcessingMetrics | None, dict[str, Any]]:
    """Wait for OTel traces from Jaeger and return (summary, per-trace metrics)."""
    num_expected = sum(1 for r in results if r.success)
    otel_metrics = None
    trace_metrics_by_id: dict[str, Any] = {}
    try:
        # Quick connectivity check — if Jaeger API is unreachable, fail hard
        # so the entire schedule aborts rather than silently proceeding
        # without OTel data.
        try:
            await otel_processor._get_traces(limit=1, lookback="1m")
        except (aiohttp.ClientConnectorError, aiohttp.ClientError, OSError) as e:
            raise RuntimeError(
                f"Jaeger API is unreachable at {otel_processor.jaeger_url}: {e}"
            ) from e

        print(
            f"Waiting for OTel traces to arrive at Jaeger (expecting {num_expected})..."
        )
        traces: list[dict[str, Any]] = []
        for _ in range(12):
            await asyncio.sleep(5)
            traces = await otel_processor._get_traces(
                limit=num_expected + 100, lookback="1h"
            )
            print(f"  {len(traces)}/{num_expected} traces available...")
            if len(traces) >= num_expected:
                break

        trace_metrics_by_id = await asyncio.to_thread(
            _process_traces_sync, otel_processor, traces
        )

        otel_metrics = await asyncio.to_thread(
            _save_and_summarize_traces_sync, otel_processor, traces, otel_output_dir
        )
        print(
            f"OTel: {otel_metrics.total_requests} requests, "
            f"avg latency: {otel_metrics.avg_latency_ms:.2f}ms, "
            f"avg queuing: {otel_metrics.avg_queuing_delay_ms:.2f}ms "
            f"({otel_metrics.queuing_delay_percentage:.1f}%)"
        )
    except RuntimeError:
        raise  # Jaeger unreachable — abort the entire schedule
    except Exception as e:
        print(f"Warning: Failed to collect OTel traces: {e}")
        print("Benchmark results will be saved without OTel metrics.")

    return otel_metrics, trace_metrics_by_id


# ---------------------------------------------------------------------------
# Metrics and result building
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {
            k: 0.0 for k in ["mean", "median", "std", "p90", "p95", "p99", "min", "max"]
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _build_data(
    app: Any,
    results: list[RequestResult],
    benchmark_duration: float,
    trace_metrics_by_id: dict[str, Any],
    otel_metrics: ProcessingMetrics | None,
    app_id: str,
) -> dict[str, Any]:
    """Map OTel traces to results, compute metrics, build the saved data dict."""
    # Map OTel queuing delays onto result objects
    for r in results:
        if r.trace_id and r.trace_id in trace_metrics_by_id:
            r.queuing_delay = trace_metrics_by_id[r.trace_id].queuing_delay_ms / 1000.0

    mapped = sum(1 for r in results if r.trace_id and r.trace_id in trace_metrics_by_id)
    if trace_metrics_by_id:
        print(f"Mapped {mapped}/{len(results)} results to OTel traces")

    # Build per-request OTel latencies (server-side span timing)
    otel_e2el_values: list[float] = []
    otel_e2el_nq_values: list[float] = []
    otel_by_trace: dict[str, dict[str, float]] = {}
    for r in results:
        if r.trace_id and r.trace_id in trace_metrics_by_id:
            tm = trace_metrics_by_id[r.trace_id]
            span_duration_ms = tm.e2e_latency_ms + tm.queuing_delay_ms
            otel_by_trace[r.trace_id] = {
                "otel_e2el_ms": span_duration_ms,
                "otel_e2el_no_queue_ms": tm.e2e_latency_ms,
            }
            if r.success:
                otel_e2el_values.append(span_duration_ms)
                otel_e2el_nq_values.append(tm.e2e_latency_ms)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    throughput = len(successful) / benchmark_duration if benchmark_duration > 0 else 0.0
    is_mllm = isinstance(
        app,
        (
            DummyLLMApp,
            MonolithicLLMApp,
            OmniMLLMApp,
            OmniRouterApp,
            OmniFlexApp,
            ModServeApp,
            TimeSharingApp,
            MLLMRouterApp,
            MixedMLLMApp,
            GroupedMixedMLLMApp,
            DummyTalkerVocoderApp,
            PrefillLLMApp,
            DecodeLLMApp,
        ),
    )

    # Base metrics (all app types)
    data: dict[str, Any] = {
        "app_id": app_id,
        "throughput": throughput,
        "benchmark_duration": benchmark_duration,
        "num_successful": len(successful),
        "num_failed": len(failed),
        "e2el_ms": _stats([r.latency * 1000 for r in successful]),
        "e2el_no_queue_ms": _stats(
            [(r.latency - r.queuing_delay) * 1000 for r in successful]
        ),
    }

    if is_mllm:
        ttfts = [r.ttft * 1000 for r in successful if r.ttft > 0]
        ttfts_nq = [(r.ttft - r.queuing_delay) * 1000 for r in successful if r.ttft > 0]
        all_itls = [itl * 1000 for r in successful for itl in r.itl]
        total_output = sum(r.output_tokens for r in successful)
        total_input = sum(r.prompt_len for r in successful)
        data.update(
            {
                "output_throughput": total_output / benchmark_duration
                if benchmark_duration > 0
                else 0.0,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "ttft_ms": _stats(ttfts),
                "ttft_no_queue_ms": _stats(ttfts_nq),
                "itl_ms": _stats(all_itls),
            }
        )
        if isinstance(
            app,
            (
                ModServeApp,
                TimeSharingApp,
                MLLMRouterApp,
                MixedMLLMApp,
                GroupedMixedMLLMApp,
            ),
        ):
            eric_q = [
                trace_metrics_by_id[r.trace_id].eric_queuing_delay_ms
                for r in successful
                if r.trace_id and r.trace_id in trace_metrics_by_id
            ]
            eric_proc = [
                trace_metrics_by_id[r.trace_id].eric_latency_ms
                for r in successful
                if r.trace_id and r.trace_id in trace_metrics_by_id
            ]
            llm_q = [
                trace_metrics_by_id[r.trace_id].llm_queuing_delay_ms
                for r in successful
                if r.trace_id and r.trace_id in trace_metrics_by_id
            ]
            llm_proc = [
                trace_metrics_by_id[r.trace_id].llm_latency_ms
                for r in successful
                if r.trace_id and r.trace_id in trace_metrics_by_id
            ]
            data.update(
                {
                    "eric_queuing_ms": _stats(eric_q),
                    "eric_processing_ms": _stats(eric_proc),
                    "llm_queuing_ms": _stats(llm_q),
                    "llm_processing_ms": _stats(llm_proc),
                }
            )
    else:
        engine_queuing = [
            trace_metrics_by_id[r.trace_id].queuing_delay_ms
            for r in results
            if r.trace_id and r.trace_id in trace_metrics_by_id
        ]
        data["engine_queuing_ms"] = _stats(engine_queuing)

    data["otel_metrics"] = otel_metrics.model_dump() if otel_metrics else {}
    data["otel_e2el_ms"] = _stats(otel_e2el_values)
    data["otel_e2el_no_queue_ms"] = _stats(otel_e2el_nq_values)

    # Per-result serialization
    def _result_dict(r: RequestResult) -> dict[str, Any]:
        otel_data = otel_by_trace.get(r.trace_id) if r.trace_id else None
        d: dict[str, Any] = {
            "trace_id": r.trace_id,
            "success": r.success,
            "latency": r.latency,
            "send_time": r.send_time,
            "queuing_delay": r.queuing_delay,
            "latency_no_queue": r.latency - r.queuing_delay,
            "otel_e2el_ms": otel_data["otel_e2el_ms"] if otel_data else None,
            "otel_e2el_no_queue_ms": otel_data["otel_e2el_no_queue_ms"]
            if otel_data
            else None,
            "error": r.error,
        }
        if is_mllm:
            d.update(
                {
                    "ttft": r.ttft,
                    "ttft_no_queue": r.ttft - r.queuing_delay,
                    "itl": r.itl,
                    "output_tokens": r.output_tokens,
                    "prompt_len": r.prompt_len,
                }
            )
            if r.image_resolutions:
                d["image_resolutions"] = r.image_resolutions
            # Omni-specific modality info
            if r.return_audio or r.num_images or r.num_videos or r.num_audios:
                d["return_audio"] = r.return_audio
                d["num_images"] = r.num_images
                d["num_videos"] = r.num_videos
                d["num_audios"] = r.num_audios
                if r.video_metadata:
                    d["video_metadata"] = r.video_metadata
                if r.audio_durations_s:
                    d["audio_durations_s"] = r.audio_durations_s
            if r.trace_id and r.trace_id in trace_metrics_by_id:
                tm = trace_metrics_by_id[r.trace_id]
                if hasattr(tm, "mm_items"):
                    d["mm_items"] = [item.model_dump() for item in tm.mm_items]
                if hasattr(tm, "eric_queuing_delay_ms"):
                    d["eric_queuing_delay_ms"] = tm.eric_queuing_delay_ms
                    d["eric_latency_ms"] = tm.eric_latency_ms
                    d["llm_queuing_delay_ms"] = tm.llm_queuing_delay_ms
                    d["llm_latency_ms"] = tm.llm_latency_ms
                    d["num_prompt_tokens"] = tm.num_prompt_tokens
                    d["num_output_tokens"] = tm.num_output_tokens
                if hasattr(tm, "talker_latency_ms"):
                    d["talker_queuing_delay_ms"] = tm.talker_queuing_delay_ms
                    d["talker_latency_ms"] = tm.talker_latency_ms
                    d["talker_output_tokens"] = tm.talker_output_tokens
        else:
            if r.trace_id and r.trace_id in trace_metrics_by_id:
                tm = trace_metrics_by_id[r.trace_id]
                for k, v in tm.model_dump().items():
                    if k not in ("trace_id", "e2e_latency_ms", "queuing_delay_ms"):
                        d[k] = v
        return d

    data["results"] = [_result_dict(r) for r in results]

    print(
        f"Benchmark done: {len(successful)}/{len(results)} successful, {benchmark_duration:.1f}s"
    )
    if failed:
        for r in failed[:3]:
            print(f"  Error: {r.error[:200]}")

    return data


# ---------------------------------------------------------------------------
# Unified benchmark entry point
# ---------------------------------------------------------------------------


class BenchmarkSession:
    """Manages the state and phases of a single experiment benchmark.

    Splits the monolithic ``benchmark()`` flow into four discrete phases
    (``setup``, ``dispatch``, ``collect``, ``teardown``) so that the
    batch scheduler can synchronize multiple experiments at phase
    boundaries.
    """

    def __init__(
        self,
        experiment: Experiment,
        app_id: str,
        request_rate: float = float("inf"),
        max_concurrency: int | None = None,
        inputs: list[Any] | None = None,
    ):
        self.experiment = experiment
        self.app = experiment.app
        self.app_id = app_id
        self.request_rate = request_rate
        self.max_concurrency = max_concurrency
        self._provided_inputs = inputs
        self.rng = np.random.default_rng(experiment.seed)

        # Populated during setup
        self.invoke_fn: Callable[..., Awaitable[RequestResult]] | None = None
        self.inputs: list[Any] | None = None
        self.output_dir: Path | None = None
        self.pod_prefixes: list[str] = []
        self.log_streamer: ExecutorLogStreamer | None = None
        self.port_fwd: PortForwardManager | None = None

        # Populated during dispatch
        self.results: list[RequestResult] | None = None
        self.benchmark_duration: float = 0.0
        self.benchmark_start: float = 0.0
        self.benchmark_start_wall: float = 0.0
        self.tracker: RequestTracker | None = None
        self.prom_pod_names: list[str] = []  # pod name per Prometheus collector
        self.prom_timelines: list[list[dict]] = []
        self.pod_died: bool = False  # set if a pod death is detected
        self.dead_pods: list[str] = []  # names of pods that died

        # Populated during collect
        self.data: dict[str, Any] | None = None

    async def sample_inputs(self) -> None:
        """Sample and transform benchmark inputs (CPU-bound, no app readiness needed)."""
        app = self.app
        app_id = self.app_id
        inputs = self._provided_inputs

        if isinstance(app, DummyEricApp):
            self.invoke_fn = invoke_eric
            if inputs is None:
                requests = await asyncio.to_thread(
                    self.experiment.dataset.sample, app.model_id
                )
                inputs = to_eric_inputs(requests, app_id, app.model_id)
                print(f"Sampled and transformed {len(inputs)} Eric inputs")
        elif isinstance(app, (OmniRouterApp, OmniFlexApp)):
            if inputs is None:
                requests = await asyncio.to_thread(
                    self.experiment.dataset.sample, app.model_id
                )
                if requests and isinstance(requests[0], OmniRequest):
                    self.invoke_fn = invoke_omni
                    inputs = to_omni_inputs(requests, app_id, app.model_id)
                    print(f"Sampled and transformed {len(inputs)} Omni inputs")
                else:
                    self.invoke_fn = invoke_mllm
                    inputs = to_mllm_inputs(requests, app_id, app.model_id)
                    print(f"Sampled and transformed {len(inputs)} MLLM inputs")
            else:
                if isinstance(inputs[0], OmniMLLMInput):
                    self.invoke_fn = invoke_omni
                else:
                    self.invoke_fn = invoke_mllm
        elif isinstance(
            app,
            (
                DummyLLMApp,
                MonolithicLLMApp,
                OmniMLLMApp,
                ModServeApp,
                TimeSharingApp,
                MLLMRouterApp,
                MixedMLLMApp,
                GroupedMixedMLLMApp,
                PrefillLLMApp,
                DecodeLLMApp,
                PDMLLMApp,
                EPDMLLMApp,
            ),
        ):
            if inputs is None:
                requests = await asyncio.to_thread(
                    self.experiment.dataset.sample, app.model_id
                )
                # OmniRequest has video/audio/return_audio metadata.
                # Use to_omni_inputs + invoke_omni to preserve metadata in
                # result.json.  The extra "return_audio" field in the payload
                # is ignored by non-OmniRouter backends (DummyLLM, MonoLLM,
                # OmniMLLM) since vLLM's OpenAI API ignores unknown fields.
                if requests and isinstance(requests[0], OmniRequest):
                    self.invoke_fn = invoke_omni
                    inputs = to_omni_inputs(requests, app_id, app.model_id)
                    print(f"Sampled and transformed {len(inputs)} Omni inputs")
                else:
                    self.invoke_fn = invoke_mllm
                    inputs = to_mllm_inputs(requests, app_id, app.model_id)
                    print(f"Sampled and transformed {len(inputs)} MLLM inputs")
            else:
                # inputs already provided — infer invoke_fn from type
                if isinstance(inputs[0], OmniMLLMInput):
                    self.invoke_fn = invoke_omni
                else:
                    self.invoke_fn = invoke_mllm
        elif isinstance(app, DummyImageGeriApp):
            self.invoke_fn = invoke_geri
            if inputs is None:
                ds = self.experiment.dataset
                assert isinstance(ds, ControlledDiT), (
                    f"Image geri requires ControlledDiT dataset, got {type(ds).__name__}"
                )
                url = f"{GATEWAY_URL}/app/invoke/{app_id}"
                requests = await asyncio.to_thread(ds.sample, app.model_id)
                inputs = [
                    GeriInput(
                        url=url,
                        request_data={
                            "dummy_seq_len": req.prompt_len,
                            "height": req.output_image_height,
                            "width": req.output_image_width,
                            "num_inference_steps": req.num_inference_steps,
                        },
                    )
                    for req in requests
                ]
                print(f"Built {len(inputs)} Geri image inputs")
        elif isinstance(app, (QwenImageDisaggApp, MixedQwenImageDisaggApp,
                               QwenImageTextGeriApp, MixedMonoQwenImageApp,
                               SuperGroupQwenImageApp)):
            self.invoke_fn = invoke_geri
            if inputs is None:
                ds = self.experiment.dataset
                assert isinstance(ds, ControlledDiT), (
                    f"Qwen image requires ControlledDiT dataset, got {type(ds).__name__}"
                )
                url = f"{GATEWAY_URL}/app/invoke/{app_id}"
                requests = await asyncio.to_thread(ds.sample, app.model_id)
                inputs = [
                    GeriInput(
                        url=url,
                        request_data={
                            "prompt": req.prompt,
                            "height": req.output_image_height,
                            "width": req.output_image_width,
                            "num_inference_steps": req.num_inference_steps,
                        },
                    )
                    for req in requests
                ]
                print(f"Built {len(inputs)} Qwen image inputs")
        elif isinstance(app, DummyAudioGeriApp):
            self.invoke_fn = invoke_geri
            if inputs is None:
                requests = await asyncio.to_thread(
                    self.experiment.dataset.sample, app.model_id
                )
                inputs = to_geri_audio_inputs(requests, app_id)
                print(f"Sampled and transformed {len(inputs)} Geri audio inputs")
        elif isinstance(app, DummyTalkerApp):
            self.invoke_fn = invoke_talker
            if inputs is None:
                requests = await asyncio.to_thread(
                    self.experiment.dataset.sample, app.model_id
                )
                inputs = to_talker_inputs(requests, app_id, app.model_id)
                print(f"Sampled and transformed {len(inputs)} Talker inputs")
        elif isinstance(app, DummyTalkerVocoderApp):
            self.invoke_fn = invoke_talker_vocoder
            if inputs is None:
                requests = await asyncio.to_thread(
                    self.experiment.dataset.sample, app.model_id
                )
                inputs = to_talker_inputs(requests, app_id, app.model_id)
                print(f"Sampled and transformed {len(inputs)} Talker Vocoder inputs")
        else:
            raise ValueError(f"Unsupported app type: {type(app).__name__}")

        # Clamp output_len if the experiment requests it (e.g. prefill profiling).
        if inputs and self.experiment.max_output_len is not None:
            cap = self.experiment.max_output_len
            for inp in inputs:
                if hasattr(inp, "output_len"):
                    inp.output_len = min(inp.output_len, cap)

        self.inputs = inputs

    async def setup(self) -> None:
        """Phase 1: Start log streaming, port-forwarding, and test request.

        Requires ``sample_inputs()`` to have been called first.
        """
        assert self.inputs is not None and self.invoke_fn is not None, (
            "sample_inputs() must be called before setup()"
        )
        app = self.app

        # Per-experiment output directory
        self.output_dir = self.experiment.to_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute pod prefixes for log and Prometheus filtering
        self.pod_prefixes = executor_pod_prefixes(app)
        self.log_streamer = await ExecutorLogStreamer.start(
            self.output_dir, pod_name_prefixes=self.pod_prefixes
        )

        # Start port forwarding for LLM executor pods (vLLM Prometheus metrics)
        prom_prefixes = prometheus_pod_prefixes(app)
        if prom_prefixes:
            self.port_fwd = PortForwardManager()
            await self.port_fwd.start(pod_name_prefixes=prom_prefixes)

        # Test request
        print("Testing a single request to verify setup...")
        test_result = await self.invoke_fn(self.inputs[0])
        if not test_result.success:
            raise RuntimeError(f"Test request failed: {test_result.error}")
        print(f"Test request succeeded (latency={test_result.latency * 1000:.1f}ms)")

    async def dispatch(self) -> None:
        """Phase 2: Dispatch all benchmark requests and collect raw results.

        Starts Prometheus collection, dispatches requests, stops Prometheus.
        """
        assert self.inputs is not None and self.invoke_fn is not None, (
            "setup() must be called first"
        )
        app = self.app
        inputs = self.inputs

        has_timestamps = getattr(inputs[0], "timestamp", None) is not None
        if has_timestamps:
            print(f"Dispatching {len(inputs)} requests (timestamp-based)...")
        else:
            print(
                f"Dispatching {len(inputs)} requests at {self.request_rate} req/s (Poisson)..."
            )

        # Create a RequestTracker for MLLM apps to detect steady state.
        if isinstance(
            app,
            (
                DummyLLMApp,
                MonolithicLLMApp,
                OmniMLLMApp,
                OmniRouterApp,
                OmniFlexApp,
                ModServeApp,
                TimeSharingApp,
                MLLMRouterApp,
                MixedMLLMApp,
                GroupedMixedMLLMApp,
                PrefillLLMApp,
                DecodeLLMApp,
                PDMLLMApp,
                EPDMLLMApp,
            ),
        ):
            if isinstance(app, OmniFlexApp):
                max_num_seqs = sum(
                    t.llm_num_replicas * t.llm_max_num_seqs
                    for group in app.groups
                    for t in group.thinkers
                )
            elif isinstance(app, OmniRouterApp):
                max_num_seqs = sum(
                    route.llm_num_replicas * route.llm_max_num_seqs
                    for route in app.routes
                )
            elif isinstance(app, (ModServeApp, TimeSharingApp)):
                max_num_seqs = app.num_llms * app.llm_max_num_seqs
            elif isinstance(app, (MLLMRouterApp, MixedMLLMApp)):
                total = 0
                for route in app.routes:
                    if isinstance(route, MLLMRouterMixedRouteConfig):
                        total += sum(
                            lc.llm_num_replicas * lc.llm_max_num_seqs
                            for lc in route.llm_configs
                        )
                    else:
                        total += route.llm_num_replicas * route.llm_max_num_seqs
                max_num_seqs = total
            elif isinstance(app, GroupedMixedMLLMApp):
                max_num_seqs = sum(
                    route.llm_num_replicas * route.llm_max_num_seqs
                    for group in app.groups
                    for route in group.routes
                )
            elif isinstance(app, (PDMLLMApp, EPDMLLMApp)):
                max_num_seqs = (
                    app.num_decode_replicas * app.decode_max_num_seqs
                )
            else:
                max_num_seqs = app.max_num_seqs  # type: ignore[union-attr]
            self.tracker = RequestTracker(
                max_num_seqs=max_num_seqs, num_requests=len(inputs)
            )
            print(
                f"Steady-state tracker: max_num_seqs={max_num_seqs}, num_requests={len(inputs)}"
            )

        # Start Prometheus collection for pods with vLLM metrics.
        prom_stop_event: asyncio.Event | None = None
        prom_collect_tasks: list[asyncio.Task[list[dict]]] = []
        collectors: list[PrometheusCollector] = []
        if self.port_fwd is not None:
            pod_urls = self.port_fwd.pod_urls
            if pod_urls:
                prom_stop_event = asyncio.Event()
                self.prom_pod_names = [pod_name for pod_name, _url in pod_urls]
                collectors = [
                    PrometheusCollector(url, interval=1.0)
                    for _pod_name, url in pod_urls
                ]
                prom_collect_tasks = [
                    asyncio.create_task(c.collect(prom_stop_event)) for c in collectors
                ]
                print(
                    f"Prometheus: collecting from {len(pod_urls)} executor pod(s): {self.prom_pod_names}"
                )

        self.benchmark_start = time.perf_counter()
        self.benchmark_start_wall = time.time()

        # Monitor for dead pods in the background while requests are dispatched.
        pod_cancel_event = asyncio.Event()

        async def _monitor_pod_health() -> None:
            """Periodically check for dead port-forward procs and collector failures."""
            while not self.pod_died:
                await asyncio.sleep(5.0)
                # Check port-forward subprocess health.
                if self.port_fwd is not None:
                    dead_pf = self.port_fwd.check_health()
                    for pod_name in dead_pf:
                        if pod_name not in self.dead_pods:
                            self.dead_pods.append(pod_name)
                            self.pod_died = True
                            print(
                                f"\n*** ERROR: port-forward died for pod {pod_name} "
                                f"— pod likely crashed ***"
                            )
                # Check Prometheus collector dead events.
                for i, c in enumerate(collectors):
                    if c.dead_event.is_set():
                        pod_name = self.prom_pod_names[i] if i < len(self.prom_pod_names) else f"collector-{i}"
                        if pod_name not in self.dead_pods:
                            self.dead_pods.append(pod_name)
                            self.pod_died = True
                            print(
                                f"\n*** ERROR: pod {pod_name} appears dead "
                                f"(consecutive Prometheus failures) ***"
                            )
                if self.pod_died:
                    pod_cancel_event.set()

        monitor_task = asyncio.create_task(_monitor_pod_health())

        try:
            self.results = await dispatch_requests(
                inputs,
                self.invoke_fn,
                request_rate=self.request_rate,
                max_concurrency=self.max_concurrency,
                rng=self.rng,
                tracker=self.tracker,
                benchmark_start=self.benchmark_start,
                cancel_event=pod_cancel_event,
            )
        except RuntimeError as e:
            if self.pod_died:
                # Pod died mid-dispatch — store empty results so collect()
                # can still run and save a result.json with pod_failure=True.
                print(f"\n*** Dispatch aborted: {e} ***")
                self.results = self.results or []
            else:
                raise
        finally:
            self.benchmark_duration = time.perf_counter() - self.benchmark_start

            # Stop the health monitor.
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Stop Prometheus collection.
            if prom_stop_event is not None:
                prom_stop_event.set()
            if prom_collect_tasks:
                self.prom_timelines = list(await asyncio.gather(*prom_collect_tasks))

    async def collect(self) -> dict[str, Any]:
        """Phase 3: Collect OTel traces, compute metrics, save results.

        Returns the data dict for this experiment.
        """
        assert self.results is not None, "dispatch() must be called first"
        assert self.output_dir is not None
        app = self.app

        # OTel traces
        otel_processor = _make_otel_processor(app, self.app_id)
        otel_output_dir = self.output_dir / "otel"
        otel_metrics, trace_metrics_by_id = await _collect_otel(
            self.results, otel_processor, otel_output_dir
        )

        # Metrics + save
        data = await asyncio.to_thread(
            _build_data,
            app,
            self.results,
            self.benchmark_duration,
            trace_metrics_by_id,
            otel_metrics,
            self.app_id,
        )

        # Attach steady-state metrics for MLLM benchmarks
        tracker = self.tracker
        benchmark_start = self.benchmark_start
        if tracker is not None and tracker.steady_state_duration > 0:
            data["steady_state"] = {
                "start": tracker.steady_state_start - benchmark_start,
                "end": tracker.steady_state_end - benchmark_start,
                "duration_s": tracker.steady_state_duration,
                "tokens_generated": tracker.steady_state_tokens,
                "requests_completed": tracker.steady_state_requests,
                "token_throughput": tracker.steady_state_tokens
                / tracker.steady_state_duration,
                "request_throughput": tracker.steady_state_requests
                / tracker.steady_state_duration,
            }
            print(
                f"Steady state: {tracker.steady_state_duration:.1f}s, "
                f"{tracker.steady_state_tokens} tokens, "
                f"{tracker.steady_state_requests} requests completed, "
                f"{tracker.steady_state_tokens / tracker.steady_state_duration:.1f} tok/s"
            )

            # Tag per-request results with steady-state membership.
            # A request is in SS if it was scheduled (dequeued) after ss_start
            # and completed before ss_end.
            ss_start = data["steady_state"]["start"]
            ss_end = data["steady_state"]["end"]
            ss_count = 0
            for r_dict in data["results"]:
                send_t = r_dict.get("send_time", 0.0)
                schedule_t = send_t + r_dict.get("queuing_delay", 0.0)
                end_t = send_t + r_dict.get("latency", 0.0)
                is_ss = r_dict.get("success", False) and schedule_t >= ss_start and end_t <= ss_end
                r_dict["steady_state"] = is_ss
                if is_ss:
                    ss_count += 1
            data["steady_state"]["num_steady_state_requests"] = ss_count
            print(f"Steady-state requests: {ss_count}/{len(data['results'])}")

        # Prometheus server-side metrics for MLLM benchmarks
        if (
            self.prom_timelines
            and tracker is not None
            and tracker.steady_state_duration > 0
        ):
            merged_timeline = sorted(
                [snap for tl in self.prom_timelines for snap in tl],
                key=lambda s: s["timestamp"],
            )

            ss_start_wall = self.benchmark_start_wall + (
                tracker.steady_state_start - benchmark_start
            )
            ss_end_wall = self.benchmark_start_wall + (
                tracker.steady_state_end - benchmark_start
            )

            prom_stats = await asyncio.to_thread(
                calculate_steady_state_stats,
                merged_timeline,
                steady_start=ss_start_wall,
                steady_end=ss_end_wall,
                agg_gauge_metrics={
                    "vllm:num_requests_running": "sum",
                    "vllm:num_requests_waiting": "sum",
                    "vllm:kv_cache_usage_perc": "avg",
                },
                counter_metric_names=[
                    "vllm:prompt_tokens_total",
                    "vllm:generation_tokens_total",
                ],
                histogram_metric_names=[
                    "vllm:time_to_first_token_seconds",
                    "vllm:inter_token_latency_seconds",
                    "vllm:e2e_request_latency_seconds",
                ],
            )
            data["prometheus_metrics"] = prom_stats
            print(f"Prometheus: collected {len(prom_stats)} steady-state metrics")

        # Save per-pod prometheus timelines.
        if self.prom_timelines:
            for pod_name, tl in zip(self.prom_pod_names, self.prom_timelines):
                if tl:
                    await asyncio.to_thread(
                        save_timeline, tl, self.output_dir, f"timeline-{pod_name}.json"
                    )

        # Mark pod failure if detected during dispatch.
        if self.pod_died:
            data["pod_failure"] = True
            data["dead_pods"] = self.dead_pods
            print(
                f"*** Pod failure detected: {self.dead_pods} — "
                f"results are unreliable ***"
            )

        await asyncio.to_thread(self.experiment.save, data)
        self.data = data
        return data

    async def teardown(self) -> None:
        """Phase 4: Stop port-forwarding and log streaming."""
        if self.port_fwd is not None:
            await self.port_fwd.stop()
        if self.log_streamer is not None:
            saved = await self.log_streamer.stop()
            if saved:
                print(f"Saved executor logs: {', '.join(p.name for p in saved)}")


async def benchmark(
    experiment: Experiment,
    app_id: str,
    request_rate: float = float("inf"),
    max_concurrency: int | None = None,
    inputs: list[Any] | None = None,
) -> dict[str, Any]:
    """Run a benchmark for any Cornserve app.

    Convenience wrapper that runs all four ``BenchmarkSession`` phases
    sequentially.  Determines the invoke function and OTel processor from
    the app type.  For Eric and MLLM apps, *inputs* are sampled from the
    experiment dataset.  For Geri apps, *inputs* must be provided.
    """
    session = BenchmarkSession(
        experiment, app_id, request_rate, max_concurrency, inputs
    )
    await session.sample_inputs()
    await session.setup()
    try:
        await session.dispatch()
        data = await session.collect()
        return data
    finally:
        await session.teardown()
