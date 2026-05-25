"""Prometheus metrics collection and analysis for vLLM benchmarks.

Adapted from ml-energy/mlenergy/llm/prometheus.py. Collects server-side
metrics from vLLM's ``/metrics`` endpoint via ``kubectl port-forward``,
then aggregates over the steady-state window detected by ``RequestTracker``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import aiohttp
from cornserve.services.utils import to_strict_k8s_name
from cornserve_tasklib.task.unit.llm import llm_executor_name
from schema import (
    DecodeLLMApp,
    DummyAudioGeriApp,
    DummyEricApp,
    DummyImageGeriApp,
    DummyLLMApp,
    DummyTalkerApp,
    DummyTalkerVocoderApp,
    EPDMLLMApp,
    GroupedMixedMLLMApp,
    MixedMLLMApp,
    MixedMonoQwenImageApp,
    MixedQwenImageDisaggApp,
    MLLMRouterApp,
    MLLMRouterMixedRouteConfig,
    MLLMRouterRouteConfig,
    ModServeApp,
    MonolithicLLMApp,
    OmniApp,
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

logger = logging.getLogger(__name__)

CORNSERVE_NAMESPACE = "cornserve"
VLLM_CONTAINER_PORT = 8000


# ---------------------------------------------------------------------------
# PrometheusCollector
# ---------------------------------------------------------------------------


class PrometheusCollector:
    """Polls a vLLM ``/metrics`` endpoint and stores timestamped snapshots.

    Sets :attr:`dead_event` after *max_consecutive_failures* consecutive
    poll failures, indicating the target pod is likely dead.
    """

    def __init__(
        self,
        metrics_url: str,
        interval: float = 1.0,
        max_consecutive_failures: int = 10,
    ) -> None:
        self.metrics_url = metrics_url
        self.interval = interval
        self.max_consecutive_failures = max_consecutive_failures
        self.dead_event = asyncio.Event()

    async def collect(self, stop_event: asyncio.Event) -> list[dict[str, Any]]:
        """Collect metrics periodically until *stop_event* is set.

        Returns a list of ``{"timestamp": float, "metrics": str}`` dicts
        where *timestamp* is ``time.time()`` (wall-clock).
        """
        timeline: list[dict[str, Any]] = []
        consecutive_failures = 0
        logger.info(
            "Starting Prometheus collection from %s (interval=%.1fs)",
            self.metrics_url,
            self.interval,
        )

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10.0)
        ) as session:
            while not stop_event.is_set():
                try:
                    async with session.get(self.metrics_url) as resp:
                        now = time.time()
                        if resp.status == 200:
                            metrics_text = await resp.text()
                            timeline.append({"timestamp": now, "metrics": metrics_text})
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            logger.warning(
                                "Failed to fetch metrics: HTTP %d", resp.status
                            )
                except asyncio.TimeoutError:
                    # Timeouts are transient (slow GC, network blip) — do NOT
                    # count toward pod-death detection.
                    logger.warning("Timeout fetching Prometheus metrics")
                except (
                    aiohttp.ClientConnectorError,
                    aiohttp.ServerDisconnectedError,
                    ConnectionRefusedError,
                    ConnectionResetError,
                    OSError,
                ) as e:
                    # Connection-level errors indicate the pod/port-forward
                    # is gone — count toward pod-death threshold.
                    consecutive_failures += 1
                    logger.warning(
                        "Connection error collecting Prometheus metrics: %s", e
                    )
                except Exception as e:
                    # Unexpected errors — log but don't count as pod death.
                    logger.warning("Error collecting Prometheus metrics: %s", e)

                if (
                    consecutive_failures >= self.max_consecutive_failures
                    and not self.dead_event.is_set()
                ):
                    logger.error(
                        "Pod appears dead: %d consecutive connection failures "
                        "for %s",
                        consecutive_failures,
                        self.metrics_url,
                    )
                    self.dead_event.set()
                    break

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass

        logger.info(
            "Stopped Prometheus collection. Collected %d snapshots",
            len(timeline),
        )
        return timeline


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------


def parse_gauge(metrics_text: str, metric_name: str) -> dict[str, float]:
    """Parse gauge metric values from Prometheus text format.

    Returns a dict mapping label strings to float values.
    """
    pattern = rf"^{re.escape(metric_name)}\{{([^}}]*)\}}\s+([\d.eE+-]+)"
    return {
        m.group(1): float(m.group(2))
        for m in re.finditer(pattern, metrics_text, re.MULTILINE)
    }


def parse_counter(metrics_text: str, metric_name: str) -> dict[str, float]:
    """Parse counter metric values (same format as gauges)."""
    return parse_gauge(metrics_text, metric_name)


def parse_histogram(metrics_text: str, metric_name: str) -> dict[str, Any]:
    """Parse histogram metric from Prometheus text format.

    Returns ``{"buckets": {...}, "sum": {...}, "count": {...}}``.
    """
    buckets: dict[str, float] = {}
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}

    bucket_pattern = rf"^{re.escape(metric_name)}_bucket\{{([^}}]*)\}}\s+([\d.eE+-]+)"
    sum_pattern = rf"^{re.escape(metric_name)}_sum\{{([^}}]*)\}}\s+([\d.eE+-]+)"
    count_pattern = rf"^{re.escape(metric_name)}_count\{{([^}}]*)\}}\s+([\d.eE+-]+)"

    for m in re.finditer(bucket_pattern, metrics_text, re.MULTILINE):
        buckets[m.group(1)] = float(m.group(2))
    for m in re.finditer(sum_pattern, metrics_text, re.MULTILINE):
        sums[m.group(1)] = float(m.group(2))
    for m in re.finditer(count_pattern, metrics_text, re.MULTILINE):
        counts[m.group(1)] = float(m.group(2))

    return {"buckets": buckets, "sum": sums, "count": counts}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_gauge_value(
    metrics_text: str,
    metric_name: str,
    aggregation: Literal["sum", "avg", "max"] = "max",
) -> float | None:
    """Extract a single value from a gauge metric with specified aggregation.

    When multiple entries exist (e.g. from data-parallel engines), aggregates
    them according to *aggregation*.
    """
    values = parse_gauge(metrics_text, metric_name)
    if not values:
        return None
    vals = list(values.values())
    if aggregation == "sum":
        return sum(vals)
    elif aggregation == "avg":
        return sum(vals) / len(vals)
    else:
        return max(vals)


def _calculate_histogram_percentile(
    histogram_data: dict[str, Any], percentile: float
) -> float | None:
    """Calculate a percentile from Prometheus histogram buckets."""
    buckets = histogram_data.get("buckets", {})
    if not buckets:
        return None

    bucket_data: list[tuple[float, float]] = []
    for labels, count in buckets.items():
        le_match = re.search(r'le="([^"]+)"', labels)
        if le_match:
            le_str = le_match.group(1)
            upper_bound = float("inf") if le_str == "+Inf" else float(le_str)
            bucket_data.append((upper_bound, count))

    if not bucket_data:
        return None

    bucket_data.sort(key=lambda x: x[0])
    total_count = bucket_data[-1][1]
    if total_count == 0:
        return None

    target_count = total_count * (percentile / 100.0)
    prev_upper = 0.0
    prev_count = 0.0

    for upper_bound, cumulative_count in bucket_data:
        if cumulative_count >= target_count:
            if prev_count == cumulative_count:
                return prev_upper
            bucket_width = upper_bound - prev_upper
            count_in_bucket = cumulative_count - prev_count
            fraction = (target_count - prev_count) / count_in_bucket
            return prev_upper + fraction * bucket_width
        prev_upper = upper_bound
        prev_count = cumulative_count

    return None


def _calculate_histogram_percentiles(
    histogram_data: dict[str, Any], percentiles: list[float]
) -> dict[str, float]:
    """Calculate multiple percentiles from histogram buckets."""
    results: dict[str, float] = {}
    for p in percentiles:
        value = _calculate_histogram_percentile(histogram_data, p)
        if value is not None:
            p_name = f"p{int(p)}" if p == int(p) else f"p{p}"
            results[p_name] = value
    return results


# ---------------------------------------------------------------------------
# Steady-state statistics
# ---------------------------------------------------------------------------


def calculate_steady_state_stats(
    timeline: list[dict[str, Any]],
    steady_start: float,
    steady_end: float,
    agg_gauge_metrics: dict[str, Literal["sum", "avg", "max"]],
    counter_metric_names: list[str] | None = None,
    histogram_metric_names: list[str] | None = None,
    histogram_percentiles: list[float] | None = None,
) -> dict[str, Any]:
    """Aggregate Prometheus metrics over the steady-state window.

    Args:
        timeline: Timestamped metric snapshots (wall-clock ``time.time()``).
        steady_start: Steady-state start (wall-clock).
        steady_end: Steady-state end (wall-clock).
        agg_gauge_metrics: Gauge metric names mapped to aggregation strategy.
        counter_metric_names: Counter metrics — rate is computed as
            ``(last - first) / duration``.
        histogram_metric_names: Histogram metrics — percentiles computed
            from the last snapshot's cumulative buckets.
        histogram_percentiles: Percentiles to extract (default ``[50, 90, 95, 99]``).

    Returns:
        Dict mapping metric names to computed values. Histograms produce
        entries like ``"metric_name_p50"``, ``"metric_name_p99"``, etc.
        Counters produce ``"metric_name_rate"``.
    """
    if histogram_percentiles is None:
        histogram_percentiles = [50, 90, 95, 99]

    steady_snapshots = [
        s for s in timeline if steady_start <= s["timestamp"] <= steady_end
    ]

    if not steady_snapshots:
        logger.warning(
            "No Prometheus snapshots in steady-state window (%.1f - %.1f)",
            steady_start,
            steady_end,
        )
        return {}

    logger.info(
        "Analyzing %d Prometheus snapshots during steady state",
        len(steady_snapshots),
    )

    stats: dict[str, Any] = {}

    # --- Gauges: average over window ---
    for gauge_name, aggregation in agg_gauge_metrics.items():
        values = []
        for snapshot in steady_snapshots:
            v = _get_gauge_value(snapshot["metrics"], gauge_name, aggregation)
            if v is not None:
                values.append(v)
        if values:
            avg = sum(values) / len(values)
            stats[gauge_name] = avg
            logger.info(
                "%s: %.3f (agg=%s, %d snapshots)",
                gauge_name,
                avg,
                aggregation,
                len(values),
            )
        else:
            logger.warning("No values found for gauge: %s", gauge_name)

    # --- Counters: rate = delta / duration ---
    if counter_metric_names and len(steady_snapshots) >= 2:
        first_snap = steady_snapshots[0]
        last_snap = steady_snapshots[-1]
        duration = last_snap["timestamp"] - first_snap["timestamp"]
        if duration > 0:
            for counter_name in counter_metric_names:
                first_vals = parse_counter(first_snap["metrics"], counter_name)
                last_vals = parse_counter(last_snap["metrics"], counter_name)
                if first_vals and last_vals:
                    first_total = sum(first_vals.values())
                    last_total = sum(last_vals.values())
                    rate = (last_total - first_total) / duration
                    stats[f"{counter_name}_rate"] = rate
                    logger.info(
                        "%s: %.1f/s (delta=%.0f over %.1fs)",
                        counter_name,
                        rate,
                        last_total - first_total,
                        duration,
                    )
                else:
                    logger.warning("No values found for counter: %s", counter_name)

    # --- Histograms: percentiles from last snapshot ---
    if histogram_metric_names and steady_snapshots:
        final_snap = steady_snapshots[-1]
        for hist_name in histogram_metric_names:
            hist_data = parse_histogram(final_snap["metrics"], hist_name)
            pctls = _calculate_histogram_percentiles(hist_data, histogram_percentiles)
            if pctls:
                for p_name, value in pctls.items():
                    stats[f"{hist_name}_{p_name}"] = value
                p_str = ", ".join(f"{k}={v:.3f}" for k, v in pctls.items())
                logger.info("%s: %s", hist_name, p_str)
            else:
                logger.warning("Could not calculate percentiles for: %s", hist_name)

    return stats


# ---------------------------------------------------------------------------
# PortForwardManager
# ---------------------------------------------------------------------------


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return *values* deduplicated while preserving insertion order."""
    return list(dict.fromkeys(values))


def executor_pod_prefixes(app: Any) -> list[str]:
    """Compute executor pod name prefixes for a benchmark app."""
    raws: list[str] = []
    if isinstance(app, DummyLLMApp):
        raws.append(
            llm_executor_name("dummy-vllm", app.model_id, True, app.to_profile_str())
        )
    elif isinstance(app, MonolithicLLMApp):
        raws.append(
            llm_executor_name("dummy-vllm", app.model_id, False, app.to_profile_str())
        )
    elif isinstance(app, PrefillLLMApp):
        raws.append(
            llm_executor_name("vllm", app.model_id, app.receive_embeddings, app.to_profile_str())
        )
    elif isinstance(app, DecodeLLMApp):
        raws.append(
            llm_executor_name("vllm", app.model_id, app.receive_embeddings, app.to_profile_str() + "+pc1")
        )
    elif isinstance(app, OmniMLLMApp):
        raw = "-".join([
            "omni-vllm",
            app.model_id.split("/")[-1],
            app._enc_flags_str(),
            app.to_profile_str(),
        ]).lower()
        raws.append(raw)
    elif isinstance(app, DummyEricApp):
        model_name = app.model_id.split("/")[-1]
        raws.append(f"eric-{app.modality}-{model_name}-{app.to_profile_str()}")
    elif isinstance(app, DummyImageGeriApp):
        model_name = app.model_id.split("/")[-1]
        raws.append(f"dummy-geri-image-{model_name}-{app.to_profile_str()}")
    elif isinstance(app, QwenImageDisaggApp):
        llm_profile = (
            f"tp{app.encoder_tp_size}+bs{app.encoder_max_num_seqs}"
            f"+gpu{app.encoder_gpu_memory_utilization}"
        )
        raws.append(llm_executor_name("vllm", app.encoder_model_id, False, llm_profile))
        model_name = app.model_id.split("/")[-1]
        raws.append(f"geri-image-{model_name}")
    elif isinstance(app, MixedQwenImageDisaggApp):
        llm_profile = (
            f"tp{app.encoder_tp_size}+bs{app.encoder_max_num_seqs}"
            f"+gpu{app.encoder_gpu_memory_utilization}"
        )
        raws.append(llm_executor_name("vllm", app.encoder_model_id, False, llm_profile))
        model_name = app.model_id.split("/")[-1]
        raws.append(f"geri-image-{model_name}")
    elif isinstance(app, QwenImageTextGeriApp):
        model_name = app.model_id.split("/")[-1]
        raws.append(f"geri-image-text-{model_name}")
    elif isinstance(app, MixedMonoQwenImageApp):
        model_name = app.model_id.split("/")[-1]
        raws.append(f"geri-image-text-{model_name}")
    elif isinstance(app, SuperGroupQwenImageApp):
        model_name = app.model_id.split("/")[-1]
        # SuperGroup may contain both LLM encoders and Geri generators
        for group in app.groups:
            import json as _json
            inner = _json.loads(group.inner_app_json)
            inner_type = inner.get("_app_type", "")
            if inner_type in ("QwenImageDisaggApp", "MixedQwenImageDisaggApp"):
                llm_profile = (
                    f"tp{inner.get('encoder_tp_size', 1)}"
                    f"+bs{inner.get('encoder_max_num_seqs', 256)}"
                    f"+gpu{inner.get('encoder_gpu_memory_utilization', 0.9)}"
                )
                raws.append(llm_executor_name(
                    "vllm", inner.get("encoder_model_id", ""), False, llm_profile))
                raws.append(f"geri-image-{model_name}")
            elif inner_type in ("QwenImageTextGeriApp", "MixedMonoQwenImageApp"):
                raws.append(f"geri-image-text-{model_name}")
    elif isinstance(app, DummyAudioGeriApp):
        model_name = app.model_id.split("/")[-1]
        raws.append(f"dummy-geri-audio-{model_name}-{app.to_profile_str()}")
    elif isinstance(app, DummyTalkerApp):
        model_name = app.model_id.split("/")[-1].lower().replace(".", "-")
        raws.append(f"dummy-vllm-{model_name}-talker")
    elif isinstance(app, DummyTalkerVocoderApp):
        model_name = app.model_id.split("/")[-1].lower().replace(".", "-")
        raws.append(f"dummy-vllm-{model_name}-tv")
    elif isinstance(app, (ModServeApp, TimeSharingApp)):
        model_name = app.model_id.split("/")[-1]
        eric_profile = f"tp1+maxbs{app.eric_max_batch_size}"
        raws.append(f"eric-image-{model_name}-{eric_profile}")
        llm_profile = (
            f"tp{app.llm_tp_size}+bs{app.llm_max_num_seqs}"
            f"+gpu{app.llm_gpu_memory_utilization}"
        )
        receive_embeddings = isinstance(app, ModServeApp)
        raws.append(
            llm_executor_name("vllm", app.model_id, receive_embeddings, llm_profile)
        )
    elif isinstance(app, MLLMRouterApp):
        model_name = app.model_id.split("/")[-1]
        for route in app.routes:
            if isinstance(route, MLLMRouterMixedRouteConfig):
                eric_profile = f"tp1+maxbs{route.eric_max_batch_size}"
                raws.append(f"eric-image-{model_name}-{eric_profile}")
                for lc in route.llm_configs:
                    llm_profile = (
                        f"tp{lc.llm_tp_size}+bs{lc.llm_max_num_seqs}"
                        f"+gpu{lc.llm_gpu_memory_utilization}"
                    )
                    raws.append(llm_executor_name("vllm", app.model_id, True, llm_profile))
            else:
                assert isinstance(route, MLLMRouterRouteConfig)
                if route.has_encoder_task():
                    eric_profile = f"tp1+maxbs{route.eric_max_batch_size}"
                    raws.append(f"eric-image-{model_name}-{eric_profile}")
                llm_profile = (
                    f"tp{route.llm_tp_size}+bs{route.llm_max_num_seqs}"
                    f"+gpu{route.llm_gpu_memory_utilization}"
                )
                raws.append(
                    llm_executor_name(
                        "vllm", app.model_id, route.llm_receive_embeddings(), llm_profile
                    )
                )
    elif isinstance(app, MixedMLLMApp):
        model_name = app.model_id.split("/")[-1]
        eric_profile = f"tp1+maxbs{app.eric_max_batch_size}"
        raws.append(f"eric-image-{model_name}-{eric_profile}")
        for route in app.routes:
            llm_profile = (
                f"tp{route.llm_tp_size}+bs{route.llm_max_num_seqs}"
                f"+gpu{route.llm_gpu_memory_utilization}"
            )
            raws.append(llm_executor_name("vllm", app.model_id, True, llm_profile))
    elif isinstance(app, GroupedMixedMLLMApp):
        model_name = app.model_id.split("/")[-1]
        for group in app.groups:
            eric_profile = f"tp1+maxbs{group.eric_max_batch_size}"
            raws.append(f"eric-image-{model_name}-{eric_profile}")
            for route in group.routes:
                llm_profile = (
                    f"tp{route.llm_tp_size}+bs{route.llm_max_num_seqs}"
                    f"+gpu{route.llm_gpu_memory_utilization}"
                )
                raws.append(llm_executor_name("vllm", app.model_id, True, llm_profile))
    elif isinstance(app, OmniRouterApp):
        model_name = app.model_id.split("/")[-1]
        for route in app.routes:
            for modality, n in [
                ("image", route.img_eric_num_replicas),
                ("video", route.vid_eric_num_replicas),
                ("audio", route.audio_eric_num_replicas),
            ]:
                if n > 0:
                    eric_profile = f"tp1+maxbs{route.eric_max_batch_size}"
                    raws.append(f"eric-{modality}-{model_name}-{eric_profile}")
            receive = route.route_type == "omni_mllm" and route.encoder_fission
            llm_profile = (
                f"tp{route.llm_tp_size}+bs{route.llm_max_num_seqs}"
                f"+gpu{route.llm_gpu_memory_utilization}"
            )
            raws.append(
                llm_executor_name("vllm", app.model_id, receive, llm_profile)
            )
            if route.vocoder_fission:
                if route.talker_num_replicas > 0:
                    tk_name = model_name.lower().replace(".", "-")
                    raws.append(f"vllm-{tk_name}-talker")
                if route.audio_geri_num_replicas > 0:
                    raws.append(f"geri-audio-{model_name}")
            elif route.talker_vocoder_num_replicas > 0:
                tv_name = model_name.lower().replace(".", "-")
                raws.append(f"vllm-{tv_name}-tv")
    elif isinstance(app, OmniApp):
        model_name = app.model_id.split("/")[-1]
        for modality, n in [
            ("image", app.img_eric_num_replicas),
            ("video", app.vid_eric_num_replicas),
            ("audio", app.audio_eric_num_replicas),
        ]:
            if n > 0:
                eric_profile = f"tp1+maxbs{app.eric_max_batch_size}"
                raws.append(f"eric-{modality}-{model_name}-{eric_profile}")
        llm_profile = (
            f"tp{app.llm_tp_size}+bs{app.llm_max_num_seqs}"
            f"+gpu{app.llm_gpu_memory_utilization}"
        )
        raws.append(
            llm_executor_name("vllm", app.model_id, app.encoder_fission, llm_profile)
        )
        if app.vocoder_fission:
            if app.talker_num_replicas > 0:
                tk_name = model_name.lower().replace(".", "-")
                raws.append(f"vllm-{tk_name}-talker")
            if app.audio_geri_num_replicas > 0:
                raws.append(f"geri-audio-{model_name}")
        elif app.talker_vocoder_num_replicas > 0:
            tv_name = model_name.lower().replace(".", "-")
            raws.append(f"vllm-{tv_name}-tv")
    elif isinstance(app, OmniFlexApp):
        model_name = app.model_id.split("/")[-1]
        for group in app.groups:
            for modality, n in [
                ("image", group.img_eric_num_replicas),
                ("video", group.vid_eric_num_replicas),
                ("audio", group.audio_eric_num_replicas),
            ]:
                if n > 0:
                    bs = group.eric_max_batch_sizes.get(
                        {"image": "img", "video": "vid", "audio": "audio"}[modality], 1
                    )
                    eric_profile = f"tp1+maxbs{bs}"
                    raws.append(f"eric-{modality}-{model_name}-{eric_profile}")
            for t in group.thinkers:
                llm_profile = (
                    f"tp{t.llm_tp_size}+bs{t.llm_max_num_seqs}"
                    f"+gpu{t.llm_gpu_memory_utilization}"
                )
                raws.append(
                    llm_executor_name("vllm", app.model_id, False, llm_profile)
                )
        # Shared audio output
        if app.vocoder_fission:
            if app.talker_num_replicas > 0:
                tk_name = model_name.lower().replace(".", "-")
                raws.append(f"vllm-{tk_name}-talker")
            if app.audio_geri_num_replicas > 0:
                raws.append(f"geri-audio-{model_name}")
        elif app.talker_vocoder_num_replicas > 0:
            tv_name = model_name.lower().replace(".", "-")
            raws.append(f"vllm-{tv_name}-tv")
    elif isinstance(app, EPDMLLMApp):
        model_name = app.model_id.split("/")[-1]
        eric_profile = f"tp1+maxbs{app.eric_max_batch_size}"
        raws.append(f"eric-image-{model_name}-{eric_profile}")
        prefill_profile = app.prefill_profile_str()
        raws.append(llm_executor_name("prefill", app.model_id, True, prefill_profile))
        decode_profile = app.decode_profile_str()
        raws.append(llm_executor_name("decode", app.model_id, True, decode_profile))
    elif isinstance(app, PDMLLMApp):
        prefill_profile = app.prefill_profile_str()
        raws.append(llm_executor_name("prefill", app.model_id, False, prefill_profile))
        decode_profile = app.decode_profile_str()
        raws.append(llm_executor_name("decode", app.model_id, False, decode_profile))
    else:
        raise ValueError(f"Unsupported app type for pod prefix: {type(app).__name__}")

    return [
        "te-" + to_strict_k8s_name(raw, max_len=60)
        for raw in _dedupe_preserve_order(raws)
    ]


def prometheus_pod_prefixes(app: Any) -> list[str]:
    """Return pod prefixes for executor pods that expose vLLM ``/metrics``."""
    if isinstance(app, DummyLLMApp):
        raw = llm_executor_name("dummy-vllm", app.model_id, True, app.to_profile_str())
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, MonolithicLLMApp):
        raw = llm_executor_name("dummy-vllm", app.model_id, False, app.to_profile_str())
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, PrefillLLMApp):
        raw = llm_executor_name("vllm", app.model_id, app.receive_embeddings, app.to_profile_str())
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, DecodeLLMApp):
        raw = llm_executor_name("vllm", app.model_id, app.receive_embeddings, app.to_profile_str() + "+pc1")
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, OmniMLLMApp):
        raw = "-".join([
            "omni-vllm",
            app.model_id.split("/")[-1],
            app._enc_flags_str(),
            app.to_profile_str(),
        ]).lower()
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, ModServeApp):
        llm_profile = (
            f"tp{app.llm_tp_size}+bs{app.llm_max_num_seqs}"
            f"+gpu{app.llm_gpu_memory_utilization}"
        )
        raw = llm_executor_name("vllm", app.model_id, True, llm_profile)
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, TimeSharingApp):
        llm_profile = (
            f"tp{app.llm_tp_size}+bs{app.llm_max_num_seqs}"
            f"+gpu{app.llm_gpu_memory_utilization}"
        )
        raw = llm_executor_name("vllm", app.model_id, False, llm_profile)
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, (QwenImageDisaggApp, MixedQwenImageDisaggApp)):
        llm_profile = (
            f"tp{app.encoder_tp_size}+bs{app.encoder_max_num_seqs}"
            f"+gpu{app.encoder_gpu_memory_utilization}"
        )
        raw = llm_executor_name("vllm", app.encoder_model_id, False, llm_profile)
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, (QwenImageTextGeriApp, MixedMonoQwenImageApp)):
        # Mono Geri pods don't expose vLLM /metrics
        return []
    elif isinstance(app, SuperGroupQwenImageApp):
        # SuperGroup may contain disagg inner apps with LLM encoders
        import json as _json
        raws_sg: list[str] = []
        for group in app.groups:
            inner = _json.loads(group.inner_app_json)
            inner_type = inner.get("_app_type", "")
            if inner_type in ("QwenImageDisaggApp", "MixedQwenImageDisaggApp"):
                llm_profile = (
                    f"tp{inner.get('encoder_tp_size', 1)}"
                    f"+bs{inner.get('encoder_max_num_seqs', 256)}"
                    f"+gpu{inner.get('encoder_gpu_memory_utilization', 0.9)}"
                )
                raws_sg.append(llm_executor_name(
                    "vllm", inner.get("encoder_model_id", ""), False, llm_profile))
        return [
            "te-" + to_strict_k8s_name(raw, max_len=60)
            for raw in _dedupe_preserve_order(raws_sg)
        ]
    elif isinstance(app, MLLMRouterApp):
        raws: list[str] = []
        for route in app.routes:
            if isinstance(route, MLLMRouterMixedRouteConfig):
                for lc in route.llm_configs:
                    llm_profile = (
                        f"tp{lc.llm_tp_size}+bs{lc.llm_max_num_seqs}"
                        f"+gpu{lc.llm_gpu_memory_utilization}"
                    )
                    raws.append(llm_executor_name("vllm", app.model_id, True, llm_profile))
            else:
                assert isinstance(route, MLLMRouterRouteConfig)
                llm_profile = (
                    f"tp{route.llm_tp_size}+bs{route.llm_max_num_seqs}"
                    f"+gpu{route.llm_gpu_memory_utilization}"
                )
                raws.append(
                    llm_executor_name(
                        "vllm", app.model_id, route.llm_receive_embeddings(), llm_profile
                    )
                )
        return [
            "te-" + to_strict_k8s_name(raw, max_len=60)
            for raw in _dedupe_preserve_order(raws)
        ]
    elif isinstance(app, MixedMLLMApp):
        raws: list[str] = []
        for route in app.routes:
            llm_profile = (
                f"tp{route.llm_tp_size}+bs{route.llm_max_num_seqs}"
                f"+gpu{route.llm_gpu_memory_utilization}"
            )
            raws.append(llm_executor_name("vllm", app.model_id, True, llm_profile))
        return [
            "te-" + to_strict_k8s_name(raw, max_len=60)
            for raw in _dedupe_preserve_order(raws)
        ]
    elif isinstance(app, GroupedMixedMLLMApp):
        raws: list[str] = []
        for group in app.groups:
            for route in group.routes:
                llm_profile = (
                    f"tp{route.llm_tp_size}+bs{route.llm_max_num_seqs}"
                    f"+gpu{route.llm_gpu_memory_utilization}"
                )
                raws.append(llm_executor_name("vllm", app.model_id, True, llm_profile))
        return [
            "te-" + to_strict_k8s_name(raw, max_len=60)
            for raw in _dedupe_preserve_order(raws)
        ]
    elif isinstance(app, DummyTalkerApp):
        model_name = app.model_id.split("/")[-1].lower().replace(".", "-")
        raw = f"dummy-vllm-{model_name}-talker"
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, DummyTalkerVocoderApp):
        model_name = app.model_id.split("/")[-1].lower().replace(".", "-")
        raw = f"dummy-vllm-{model_name}-tv"
        return ["te-" + to_strict_k8s_name(raw, max_len=60)]
    elif isinstance(app, OmniRouterApp):
        raws: list[str] = []
        model_name = app.model_id.split("/")[-1]
        for route in app.routes:
            receive = route.route_type == "omni_mllm" and route.encoder_fission
            llm_profile = (
                f"tp{route.llm_tp_size}+bs{route.llm_max_num_seqs}"
                f"+gpu{route.llm_gpu_memory_utilization}"
            )
            raws.append(
                llm_executor_name("vllm", app.model_id, receive, llm_profile)
            )
            # Talker/TV pods also expose vLLM /metrics
            if route.vocoder_fission and route.talker_num_replicas > 0:
                tk_name = model_name.lower().replace(".", "-")
                raws.append(f"vllm-{tk_name}-talker")
            elif route.talker_vocoder_num_replicas > 0:
                tv_name = model_name.lower().replace(".", "-")
                raws.append(f"vllm-{tv_name}-tv")
        return [
            "te-" + to_strict_k8s_name(raw, max_len=60)
            for raw in _dedupe_preserve_order(raws)
        ]
    elif isinstance(app, OmniApp):
        raws: list[str] = []
        model_name = app.model_id.split("/")[-1]
        llm_profile = (
            f"tp{app.llm_tp_size}+bs{app.llm_max_num_seqs}"
            f"+gpu{app.llm_gpu_memory_utilization}"
        )
        raws.append(
            llm_executor_name("vllm", app.model_id, app.encoder_fission, llm_profile)
        )
        if app.vocoder_fission and app.talker_num_replicas > 0:
            tk_name = model_name.lower().replace(".", "-")
            raws.append(f"vllm-{tk_name}-talker")
        elif app.talker_vocoder_num_replicas > 0:
            tv_name = model_name.lower().replace(".", "-")
            raws.append(f"vllm-{tv_name}-tv")
        return [
            "te-" + to_strict_k8s_name(raw, max_len=60)
            for raw in _dedupe_preserve_order(raws)
        ]
    elif isinstance(app, OmniFlexApp):
        raws: list[str] = []
        model_name = app.model_id.split("/")[-1]
        for group in app.groups:
            for t in group.thinkers:
                llm_profile = (
                    f"tp{t.llm_tp_size}+bs{t.llm_max_num_seqs}"
                    f"+gpu{t.llm_gpu_memory_utilization}"
                )
                raws.append(
                    llm_executor_name("vllm", app.model_id, False, llm_profile)
                )
        # Shared audio output (exposes vLLM /metrics)
        if app.vocoder_fission and app.talker_num_replicas > 0:
            tk_name = model_name.lower().replace(".", "-")
            raws.append(f"vllm-{tk_name}-talker")
        elif app.talker_vocoder_num_replicas > 0:
            tv_name = model_name.lower().replace(".", "-")
            raws.append(f"vllm-{tv_name}-tv")
    elif isinstance(app, EPDMLLMApp):
        # Eric pods do not expose vLLM /metrics; only prefill and decode do.
        raws = [
            llm_executor_name("prefill", app.model_id, True, app.prefill_profile_str()),
            llm_executor_name("decode", app.model_id, True, app.decode_profile_str()),
        ]
        return [
            "te-" + to_strict_k8s_name(raw, max_len=60)
            for raw in _dedupe_preserve_order(raws)
        ]
    elif isinstance(app, PDMLLMApp):
        raws = [
            llm_executor_name("prefill", app.model_id, False, app.prefill_profile_str()),
            llm_executor_name("decode", app.model_id, False, app.decode_profile_str()),
        ]
        return [
            "te-" + to_strict_k8s_name(raw, max_len=60)
            for raw in _dedupe_preserve_order(raws)
        ]
    else:
        return []


class PortForwardManager:
    """Manage ``kubectl port-forward`` subprocesses to task-executor pods.

    Follows the same lifecycle pattern as
    ``cornserve_utils.ExecutorLogStreamer``.

    Usage::

        pf = PortForwardManager()
        urls = await pf.start(pod_name_prefix="te-dummy-vllm-llama")
        # ... use urls ...
        await pf.stop()
    """

    def __init__(self) -> None:
        self._procs: list[tuple[str, asyncio.subprocess.Process]] = []
        self.metrics_urls: list[str] = []
        self.pod_urls: list[tuple[str, str]] = []  # (pod_name, metrics_url)
        # Saved kwargs from start() to support restart on subprocess death.
        self._namespace: str = CORNSERVE_NAMESPACE
        self._container_port: int = VLLM_CONTAINER_PORT
        # Background drain tasks so kubectl's stdout/stderr pipes never fill.
        self._drain_tasks: list[asyncio.Task] = []

    async def start(
        self,
        pod_name_prefix: str | None = None,
        pod_name_prefixes: list[str] | None = None,
        namespace: str = CORNSERVE_NAMESPACE,
        label: str = "app=task-executor",
        container_port: int = VLLM_CONTAINER_PORT,
    ) -> list[str]:
        """Discover executor pods, start port-forwards, return metrics URLs.

        Args:
            pod_name_prefix: If set, only forward pods whose name starts with
                this prefix.  Mutually exclusive with *pod_name_prefixes*.
            pod_name_prefixes: If set, forward pods matching **any** prefix.
            namespace: Kubernetes namespace.
            label: Label selector for executor pods.
            container_port: Container port to forward.

        Uses ``:container_port`` syntax so kubectl picks a free ephemeral
        local port, avoiding conflicts when multiple experiments run
        concurrently.  The assigned port is parsed from kubectl's stdout
        line ``Forwarding from 127.0.0.1:<port> -> <container_port>``.

        Returns:
            List of ``http://localhost:<port>/metrics`` URLs, one per pod.
        """
        # Normalize prefix args
        prefixes: list[str] | None = pod_name_prefixes
        if prefixes is None and pod_name_prefix is not None:
            prefixes = [pod_name_prefix]

        # Discover pods
        proc = await asyncio.create_subprocess_exec(
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            label,
            "-o",
            "jsonpath={.items[*].metadata.name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("kubectl get pods failed: %s", stderr.decode())
            return []

        pod_names = stdout.decode().split()

        # Filter by prefix(es) if specified.
        if prefixes:
            pod_names = [
                p for p in pod_names if any(p.startswith(pfx) for pfx in prefixes)
            ]

        if not pod_names:
            logger.info(
                "No task-executor pods found for port-forwarding"
                + (f" (prefix={pod_name_prefix!r})" if pod_name_prefix else "")
            )
            return []

        logger.info("Found %d task-executor pods: %s", len(pod_names), pod_names)

        # Save for restart().
        self._namespace = namespace
        self._container_port = container_port

        urls: list[str] = []
        for pod_name in pod_names:
            url = await self._spawn_one(pod_name)
            if url is None:
                continue
            urls.append(url)
            self.pod_urls.append((pod_name, url))

        self.metrics_urls = urls
        return urls

    async def _spawn_one(self, pod_name: str) -> str | None:
        """Spawn one ``kubectl port-forward`` subprocess; return its metrics URL."""
        pf_proc = await asyncio.create_subprocess_exec(
            "kubectl",
            "port-forward",
            pod_name,
            f":{self._container_port}",
            "-n",
            self._namespace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Read the first stdout line to learn the assigned local port.
        # kubectl prints: "Forwarding from 127.0.0.1:<local> -> <remote>"
        local_port: int | None = None
        assert pf_proc.stdout is not None
        try:
            line = await asyncio.wait_for(pf_proc.stdout.readline(), timeout=10.0)
            match = re.search(r"127\.0\.0\.1:(\d+)", line.decode())
            if match:
                local_port = int(match.group(1))
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for port-forward stdout from %s",
                pod_name,
            )

        if local_port is None:
            logger.warning(
                "Could not determine local port for %s; skipping",
                pod_name,
            )
            pf_proc.terminate()
            return None

        self._procs.append((pod_name, pf_proc))
        # Drain kubectl's stdout/stderr in the background so the pipes never
        # fill (kubectl prints "Handling connection for ..." per scrape; an
        # unread pipe eventually stalls the subprocess and the port-forward
        # silently dies).
        assert pf_proc.stderr is not None
        self._drain_tasks.append(
            asyncio.create_task(self._drain_stream(pf_proc.stdout, pod_name, "stdout"))
        )
        self._drain_tasks.append(
            asyncio.create_task(self._drain_stream(pf_proc.stderr, pod_name, "stderr"))
        )
        url = f"http://localhost:{local_port}/metrics"
        logger.info(
            "Port-forwarding %s:%d -> localhost:%d",
            pod_name,
            self._container_port,
            local_port,
        )
        return url

    @staticmethod
    async def _drain_stream(
        stream: asyncio.StreamReader, pod_name: str, name: str
    ) -> None:
        """Continuously read and discard a subprocess pipe so it never fills."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                # Always surface stderr (kubectl prints errors/diagnostics
                # there when a port-forward dies). Stdout is per-connection
                # chatter we don't care about — discarded but drained.
                if name == "stderr":
                    text = line.decode(errors="replace").rstrip()
                    if text:
                        logger.warning(
                            "kubectl port-forward[%s] stderr: %s",
                            pod_name,
                            text,
                        )
        except (asyncio.CancelledError, Exception):
            return

    def check_health(self) -> list[str]:
        """Return names of pods whose port-forward subprocess has died."""
        dead = []
        for pod_name, proc in self._procs:
            if proc.returncode is not None:
                dead.append(pod_name)
        return dead

    async def restart(self, pod_name: str) -> str | None:
        """Restart the port-forward for *pod_name*; return its new metrics URL.

        Removes the dead subprocess from internal tracking and updates
        ``pod_urls`` / ``metrics_urls`` with the new ephemeral port.
        Returns ``None`` if the new port-forward could not be established.
        """
        # Remove dead proc entry, if any.
        self._procs = [(n, p) for (n, p) in self._procs if n != pod_name]

        new_url = await self._spawn_one(pod_name)
        if new_url is None:
            return None

        # Update pod_urls and metrics_urls in place so external references
        # to these lists see the new URL.
        for i, (n, _old) in enumerate(self.pod_urls):
            if n == pod_name:
                self.pod_urls[i] = (pod_name, new_url)
                if i < len(self.metrics_urls):
                    self.metrics_urls[i] = new_url
                break
        else:
            self.pod_urls.append((pod_name, new_url))
            self.metrics_urls.append(new_url)
        return new_url

    async def stop(self) -> None:
        """Terminate all ``kubectl port-forward`` processes."""
        for task in self._drain_tasks:
            task.cancel()
        self._drain_tasks.clear()
        for pod_name, proc in self._procs:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            logger.info("Stopped port-forward for %s", pod_name)
        self._procs.clear()
        self.metrics_urls.clear()


# ---------------------------------------------------------------------------
# Timeline I/O
# ---------------------------------------------------------------------------


def save_timeline(
    timeline: list[dict[str, Any]],
    output_dir: Path,
    filename: str = "timeline.json",
) -> Path:
    """Save raw timeline to ``output_dir/prometheus/{filename}``.

    Strips the raw metrics text to keep only parsed values for each snapshot,
    keeping file sizes reasonable.
    """
    prom_dir = output_dir / "prometheus"
    prom_dir.mkdir(parents=True, exist_ok=True)
    out_path = prom_dir / filename

    # Store timestamps + a few key parsed metrics per snapshot instead of the
    # full text blob.  This keeps the file readable and small.
    compact: list[dict[str, Any]] = []
    for snap in timeline:
        entry: dict[str, Any] = {"timestamp": snap["timestamp"]}
        text = snap["metrics"]
        running = _get_gauge_value(text, "vllm:num_requests_running", "sum")
        waiting = _get_gauge_value(text, "vllm:num_requests_waiting", "sum")
        kv_usage = _get_gauge_value(text, "vllm:kv_cache_usage_perc", "avg")
        prompt_tok = parse_counter(text, "vllm:prompt_tokens_total")
        gen_tok = parse_counter(text, "vllm:generation_tokens_total")
        entry["num_requests_running"] = running
        entry["num_requests_waiting"] = waiting
        entry["kv_cache_usage_perc"] = kv_usage
        entry["prompt_tokens_total"] = sum(prompt_tok.values()) if prompt_tok else None
        entry["generation_tokens_total"] = sum(gen_tok.values()) if gen_tok else None
        compact.append(entry)

    with open(out_path, "w") as f:
        json.dump(compact, f, indent=2)

    logger.info(
        "Saved Prometheus timeline (%d snapshots) to %s", len(compact), out_path
    )
    return out_path
