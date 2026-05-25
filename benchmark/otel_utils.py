"""OpenTelemetry trace processing utilities for benchmark analysis."""

from __future__ import annotations

from abc import abstractmethod
import json
from pathlib import Path
from typing import Any, Literal

import aiohttp
import numpy as np
from pydantic import BaseModel


class TraceMetrics(BaseModel):
    """Processed metrics from a trace."""
    trace_id: str
    e2e_latency_ms: float  # processing time (queuing delay removed)
    queuing_delay_ms: float  # time spent waiting in scheduler queue

class EricTraceMetrics(TraceMetrics):
    """Processed metrics from an Eric trace."""
    output_dims: list[tuple[int, int]]  # List of (sequence_length, hidden_dim) for data item

class EricImageTraceMetrics(EricTraceMetrics):
    """Processed metrics from an Eric image trace."""
    image_resolutions: list[tuple[int, int]]  # List of (width, height) for each image
    num_patches: list[int]  # Number of patches for each image

class EricVideoTraceMetrics(EricTraceMetrics):
    """Processed metrics from an Eric video trace."""
    frame_resolutions: list[tuple[int, int]]  # List of (width, height) for each video's frames
    num_frames: list[int]  # Number of frames for each video
    num_patches: list[int]  # Number of patches for each video

class EricAudioTraceMetrics(EricTraceMetrics):
    """Processed metrics from an Eric audio trace."""
    audio_lengths: list[int]  # Number of samples for each audio clip
    num_tokens: list[int]  # Number of audio tokens/chunks for each clip

class GeriTraceMetrics(TraceMetrics):
    """Processed metrics from a Geri trace."""
    input_seq_len: int  # embedding sequence length (from dummy_seq_len)

class GeriImageTraceMetrics(GeriTraceMetrics):
    """Processed metrics from a Geri image trace."""
    height: int
    width: int
    num_inference_steps: int

class GeriAudioTraceMetrics(GeriTraceMetrics):
    """Processed metrics from a Geri audio trace."""


class DisaggImageTraceMetrics(GeriImageTraceMetrics):
    """Processed metrics from a disaggregated image trace (LLM encoder + Geri).

    ``queuing_delay_ms`` = ``llm_queuing_delay_ms + geri_queuing_delay_ms``
    ``e2e_latency_ms``   = ``llm_latency_ms + geri_latency_ms``
    """
    llm_queuing_delay_ms: float
    llm_latency_ms: float
    geri_queuing_delay_ms: float
    geri_latency_ms: float


class LLMMMItem(BaseModel):
    """Metadata for a single multimodal item in an LLM request."""
    modality: str
    encoder_tokens: int

class MLLMTraceMetrics(TraceMetrics):
    """Processed metrics from an MLLM trace (Eric encoder + LLM combined).

    ``queuing_delay_ms`` = ``eric_queuing_delay_ms + llm_queuing_delay_ms``
    ``e2e_latency_ms``   = ``eric_latency_ms + llm_latency_ms``
    """
    eric_queuing_delay_ms: float
    eric_latency_ms: float  # total Eric processing (queuing removed, summed across encoder spans)
    llm_queuing_delay_ms: float
    llm_latency_ms: float  # LLM processing (queuing removed)
    num_prompt_tokens: int
    num_output_tokens: int
    mm_items: list[LLMMMItem]

class OmniRouterTraceMetrics(MLLMTraceMetrics):
    """Processed metrics from an OmniRouter trace (Eric + Thinker + optional Talker).

    Extends ``MLLMTraceMetrics`` with separate talker (audio generation) fields.
    For non-audio requests the talker fields are zero.

    ``e2e_latency_ms`` and ``llm_latency_ms`` report the **thinker** only.
    ``talker_latency_ms`` reports the talker stage separately.
    """
    talker_queuing_delay_ms: float = 0.0
    talker_latency_ms: float = 0.0  # Talker processing (queuing removed)
    talker_output_tokens: int = 0
    return_audio: bool = False


class LLMTraceMetrics(TraceMetrics):
    """Processed metrics from an LLM trace."""
    num_prompt_tokens: int
    num_output_tokens: int
    mm_items: list[LLMMMItem]

class ProcessingMetrics(BaseModel):
    """Summary metrics for processed traces."""
    app_id: str
    total_traces: int
    total_requests: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_queuing_delay_ms: float
    min_queuing_delay_ms: float
    max_queuing_delay_ms: float
    avg_e2e_latency_ms: float  # avg latency + avg queuing delay
    queuing_delay_percentage: float  # queuing as % of total e2e time

class OtelProcessor:
    def __init__(self, app_id: str, jaeger_host: str = "localhost", jaeger_port: int = 30686, service_name: str = "gateway"):
        """Initialize the processor.

        Args:
            app_id: The application ID to filter traces for
            jaeger_host: Jaeger query service hostname
            jaeger_port: Jaeger query service port (default: 30686)
            service_name: OpenTelemetry service name (default: "gateway" for /app/invoke endpoints)
        """
        self.app_id = app_id
        self.jaeger_url = f"http://{jaeger_host}:{jaeger_port}"
        self.service_name = service_name

    async def _get_traces(self, limit: int = 5000, lookback: str = "12h") -> list[dict[str, Any]]:
        """Query all traces from Jaeger for the app_id.

        Uses Jaeger's server-side tag filtering to only retrieve traces
        matching our app_id, avoiding trace loss when the limit is reached
        and other apps' traces fill up the result set.

        Args:
            limit: Maximum number of traces to retrieve
            lookback: Time range to look back (e.g., "1h", "24h")

        Returns:
            List of trace objects from Jaeger
        """
        # Jaeger API endpoint for searching traces
        url = f"{self.jaeger_url}/api/traces"

        params = {
            "service": self.service_name,
            "limit": limit,
            "lookback": lookback,
            # Server-side tag filter: Jaeger will only return traces
            # with spans containing this tag, so the limit applies to
            # matching traces only (not all gateway traces).
            "tags": json.dumps({"gateway.invoke_app.app_id": self.app_id}),
        }

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

        # Traces are already filtered by app_id server-side via the tags param.
        # Apply client-side filter as a safety check.
        filtered_traces = []
        if "data" in data:
            for trace in data["data"]:
                if self._trace_matches_app_id(trace):
                    filtered_traces.append(trace)

        return filtered_traces

    def _trace_matches_app_id(self, trace: dict[str, Any]) -> bool:
        """Check if a trace matches the app_id."""
        if "spans" not in trace:
            return False

        for span in trace["spans"]:
            # Check operation name for /app/invoke pattern
            operation_name = span.get("operationName", "")
            if "/app/invoke" in operation_name:
                # Check tags for the actual app_id
                tags = span.get("tags", [])
                for tag in tags:
                    # Look for gateway.invoke_app.app_id tag
                    if tag.get("key") == "gateway.invoke_app.app_id" and tag.get("value") == self.app_id:
                        return True

        return False

    def _extract_event_timestamp(self, span: dict[str, Any], event_name: str) -> int | None:
        """Extract timestamp of a specific event from a span.

        Args:
            span: The span object
            event_name: Name of the event to find

        Returns:
            Timestamp in microseconds, or None if not found
        """
        logs = span.get("logs", [])
        for log in logs:
            fields = log.get("fields", [])
            for field in fields:
                if field.get("key") == "event" and field.get("value") == event_name:
                    return log.get("timestamp")
        return None

    def _extract_all_event_timestamps(self, span: dict[str, Any], event_name: str) -> list[int]:
        """Extract all timestamps of a specific event from a span.

        Args:
            span: The span object
            event_name: Name of the event to find

        Returns:
            List of timestamps in microseconds
        """
        timestamps = []
        logs = span.get("logs", [])
        for log in logs:
            fields = log.get("fields", [])
            for field in fields:
                if field.get("key") == "event" and field.get("value") == event_name:
                    timestamps.append(log.get("timestamp"))
                    break
        return timestamps

    @abstractmethod
    def _process_trace(self, trace: dict[str, Any]) -> list[TraceMetrics]:
        """Process a single trace to extract timing metrics."""

    def _summarize_and_save(
        self,
        traces: list[dict[str, Any]],
        output_path: Path,
    ) -> ProcessingMetrics:
        """Process traces, compute summary metrics, and save results.

        Args:
            traces: List of raw Jaeger trace dicts.
            output_path: Directory to write processed.json into.

        Returns:
            Summary statistics of the processing.
        """
        all_metrics: list[TraceMetrics] = []
        requests_list: list[dict[str, Any]] = []

        for trace in traces:
            trace_metrics = self._process_trace(trace)
            all_metrics.extend(trace_metrics)
            requests_list.extend([m.model_dump() for m in trace_metrics])

        latencies = [m.e2e_latency_ms for m in all_metrics]
        queuing_delays = [m.queuing_delay_ms for m in all_metrics]
        e2e_totals = [m.e2e_latency_ms + m.queuing_delay_ms for m in all_metrics]

        metrics = ProcessingMetrics(
            app_id=self.app_id,
            total_traces=len(traces),
            total_requests=len(all_metrics),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            min_latency_ms=min(latencies) if latencies else 0.0,
            max_latency_ms=max(latencies) if latencies else 0.0,
            median_latency_ms=float(np.percentile(latencies, 50)) if latencies else 0.0,
            p90_latency_ms=float(np.percentile(latencies, 90)) if latencies else 0.0,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            p99_latency_ms=float(np.percentile(latencies, 99)) if latencies else 0.0,
            avg_queuing_delay_ms=sum(queuing_delays) / len(queuing_delays) if queuing_delays else 0.0,
            min_queuing_delay_ms=min(queuing_delays) if queuing_delays else 0.0,
            max_queuing_delay_ms=max(queuing_delays) if queuing_delays else 0.0,
            avg_e2e_latency_ms=sum(e2e_totals) / len(e2e_totals) if e2e_totals else 0.0,
            queuing_delay_percentage=(sum(queuing_delays) / sum(e2e_totals) * 100) if e2e_totals and sum(e2e_totals) > 0 else 0.0,
        )

        output_path.mkdir(parents=True, exist_ok=True)
        processed_file = output_path / "processed.json"
        with open(processed_file, "w") as f:
            json.dump({
                "metrics": metrics.model_dump(),
                "requests": requests_list,
            }, f, indent=2)

        return metrics

    async def process_and_save(self, output_dir: Path | str, limit: int = 5000, lookback: str = "12h") -> ProcessingMetrics:
        """Query traces from Jaeger, save raw traces, process, and save results.

        Args:
            output_dir: Directory to save results.
            limit: Maximum number of traces to retrieve.
            lookback: Time range to look back.

        Returns:
            Summary statistics of the processing.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        traces = await self._get_traces(limit=limit, lookback=lookback)

        # Save all raw traces as a single Jaeger-compatible JSON file
        with open(output_path / "raw.json", "w") as f:
            json.dump({"data": traces}, f, indent=2)

        return self._summarize_and_save(traces, output_path)

    def load_and_process(self, output_dir: Path | str) -> ProcessingMetrics:
        """Load previously saved raw traces and reprocess them.

        Reads traces from ``output_dir/raw.json`` (Jaeger-compatible combined
        format), runs the processing pipeline, and overwrites
        ``output_dir/processed.json``.

        Args:
            output_dir: Directory containing ``raw.json``.

        Returns:
            Summary statistics of the reprocessed traces.
        """
        raw_path = Path(output_dir) / "raw.json"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw trace file not found: {raw_path}")

        with open(raw_path) as f:
            raw_data = json.load(f)
        traces: list[dict[str, Any]] = raw_data.get("data", [])

        return self._summarize_and_save(traces, Path(output_dir))


class EricOtelProcessor(OtelProcessor):
    """Process OpenTelemetry traces for Eric encoder requests."""

    def __init__(
        self,
        app_id: str,
        modality: Literal["image", "video", "audio"] = "image",
        jaeger_host: str = "localhost",
        jaeger_port: int = 30686,
        service_name: str = "gateway",
    ):
        """Initialize the Eric processor.

        Args:
            app_id: The application ID to filter traces for
            modality: The modality type (image, audio, video)
            jaeger_host: Jaeger query service hostname
            jaeger_port: Jaeger query service port (default: 30686)
            service_name: OpenTelemetry service name (default: "gateway" for /app/invoke endpoints)
        """
        super().__init__(app_id, jaeger_host, jaeger_port, service_name)
        self.modality = modality

    @staticmethod
    def _parse_shape(value: Any) -> list[int] | None:
        """Parse a shape value from a Jaeger span tag into a list of ints."""
        shape_str = str(value).replace("(", "[").replace(")", "]")
        if not shape_str.startswith("["):
            return None
        return eval(shape_str)

    def _extract_metadata(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Extract modality metadata from trace spans in a single pass.

        Scans three span types in one iteration:
        - ModalityDataLoader.load_from_url: raw data shapes (no data_id)
        - Processor.process: processed tensor shapes (keyed by data_id)
        - Engine._request_receive_loop: output dims (keyed by data_id)

        Returns:
            Dict of modality-specific metadata matching the fields of the
            corresponding EricTraceMetrics subclass.
        """
        # Which processor tensor key to look for per modality
        processor_key_map = {
            "image": "pixel_values_flat.shape",
            "video": "pixel_values_videos.shape",
            "audio": "input_audio_features.shape",
        }
        processor_key = processor_key_map[self.modality]

        loader_shapes: list[list[int]] = []
        processed_shapes: dict[str, list[int]] = {}  # data_id -> shape
        output_dims_dict: dict[str, tuple[int, int]] = {}  # data_id -> (seq_len, hidden_dim)

        for span in trace.get("spans", []):
            op = span.get("operationName", "")
            tags = span.get("tags", [])

            if "ModalityDataLoader.load_from_url" in op:
                for tag in tags:
                    if tag.get("key") == "data.shape":
                        shape = self._parse_shape(tag.get("value", ""))
                        if shape is not None:
                            loader_shapes.append(shape)

            elif "Processor.process" in op:
                for tag in tags:
                    key = tag.get("key", "")
                    if processor_key in key:
                        # key format: processor.processed_data.{data_id}.{tensor}.shape
                        parts = key.split(".")
                        if len(parts) >= 3:
                            data_id = parts[2]
                            shape = self._parse_shape(tag.get("value", ""))
                            if shape is not None:
                                processed_shapes[data_id] = shape

            elif "Engine._request_receive_loop" in op:
                for tag in tags:
                    key = tag.get("key", "")
                    if "worker.output." in key and ".shape" in key:
                        # key format: worker.output.{data_id}.shape
                        parts = key.split(".")
                        if len(parts) >= 3:
                            data_id = parts[2]
                            shape = self._parse_shape(tag.get("value", ""))
                            if shape is not None and len(shape) >= 2:
                                output_dims_dict[data_id] = (shape[0], shape[1])

        # Build ordered lists from data_id-keyed dicts
        sorted_ids = sorted(set(processed_shapes.keys()) | set(output_dims_dict.keys()))
        output_dims = [output_dims_dict.get(did, (0, 0)) for did in sorted_ids]

        if self.modality == "image":
            # loader shape: [H, W, 3] → (W, H)
            image_resolutions = [(s[1], s[0]) for s in loader_shapes if len(s) >= 2]
            # pixel_values_flat shape: [num_patches, C, pH, pW]
            num_patches = [processed_shapes[did][0] for did in sorted_ids if did in processed_shapes and len(processed_shapes[did]) >= 4]
            return {
                "image_resolutions": image_resolutions,
                "num_patches": num_patches,
                "output_dims": output_dims,
            }
        elif self.modality == "video":
            # loader shape: [F, H, W, 3] → (W, H), F
            frame_resolutions = [(s[2], s[1]) for s in loader_shapes if len(s) == 4]
            num_frames = [s[0] for s in loader_shapes if len(s) == 4]
            # pixel_values_videos shape: [num_patches, ...]
            num_patches = [processed_shapes[did][0] for did in sorted_ids if did in processed_shapes and len(processed_shapes[did]) >= 2]
            return {
                "frame_resolutions": frame_resolutions,
                "num_frames": num_frames,
                "num_patches": num_patches,
                "output_dims": output_dims,
            }
        elif self.modality == "audio":
            # loader shape: [num_samples]
            audio_lengths = [s[0] for s in loader_shapes if len(s) == 1]
            # input_audio_features shape: [batch, num_tokens, ...]
            num_tokens = [processed_shapes[did][1] for did in sorted_ids if did in processed_shapes and len(processed_shapes[did]) >= 2]
            return {
                "audio_lengths": audio_lengths,
                "num_tokens": num_tokens,
                "output_dims": output_dims,
            }
        else:
            raise RuntimeError(f"Unsupported modality: {self.modality}")

    def _process_trace(self, trace: dict[str, Any]) -> list[EricTraceMetrics]:
        """Process a single trace to extract timing and modality metrics.

        Args:
            trace: Trace object from Jaeger

        Returns:
            List of EricTraceMetrics (one per request in the trace)
        """
        metrics_list = []

        if "spans" not in trace:
            return metrics_list

        trace_id = trace.get("traceID", "unknown")
        modality_metadata = self._extract_metadata(trace)

        METRICS_CLS = {
            "image": EricImageTraceMetrics,
            "video": EricVideoTraceMetrics,
            "audio": EricAudioTraceMetrics,
        }
        cls = METRICS_CLS[self.modality]

        # Find Engine._request_receive_loop spans for timing.
        # Subtract queuing delay so e2e_latency_ms = processing time only.
        for span in trace["spans"]:
            if "Engine._request_receive_loop" not in span.get("operationName", ""):
                continue

            span_duration_us = span.get("duration", 0)
            enqueue_time = self._extract_event_timestamp(span, "engine.scheduler.enqueue")
            dequeue_times = self._extract_all_event_timestamps(span, "engine.scheduler.dequeue")

            if enqueue_time and dequeue_times:
                queuing_delay_us = min(dequeue_times) - enqueue_time
                queuing_delay_ms = queuing_delay_us / 1000.0
                latency_ms = (span_duration_us - queuing_delay_us) / 1000.0
            else:
                queuing_delay_ms = 0.0
                latency_ms = span_duration_us / 1000.0

            metrics_list.append(cls(
                trace_id=trace_id,
                e2e_latency_ms=latency_ms,
                queuing_delay_ms=queuing_delay_ms,
                **modality_metadata,
            ))

        return metrics_list


class GeriOtelProcessor(OtelProcessor):
    """Process OpenTelemetry traces for Geri generator requests.

    Extracts timing from ``geri.engine.process_request`` spans and computes
    queuing delay from scheduler enqueue/schedule events.
    """

    def __init__(
        self,
        app_id: str,
        modality: Literal["image", "audio"] = "image",
        jaeger_host: str = "localhost",
        jaeger_port: int = 30686,
        service_name: str = "gateway",
    ):
        super().__init__(app_id, jaeger_host, jaeger_port, service_name)
        self.modality = modality

    def _process_trace(self, trace: dict[str, Any]) -> list[GeriTraceMetrics]:
        """Process a single trace to extract timing and modality metrics."""
        metrics_list: list[GeriTraceMetrics] = []

        if "spans" not in trace:
            return metrics_list

        trace_id = trace.get("traceID", "unknown")

        for span in trace["spans"]:
            if "geri.engine.process_request" not in span.get("operationName", ""):
                continue

            tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
            span_duration_us = span.get("duration", 0)

            # Queuing delay = schedule - enqueue
            enqueue_time = self._extract_event_timestamp(span, "geri.engine.scheduler.enqueue")
            schedule_time = self._extract_event_timestamp(span, "geri.engine.scheduler.schedule")

            if enqueue_time and schedule_time:
                queuing_delay_us = schedule_time - enqueue_time
                queuing_delay_ms = queuing_delay_us / 1000.0
                latency_ms = (span_duration_us - queuing_delay_us) / 1000.0
            else:
                queuing_delay_ms = 0.0
                latency_ms = span_duration_us / 1000.0

            input_seq_len = int(tags.get("geri.engine.process_request.dummy_seq_len", 0))

            METRICS_CLS = {
                "image": GeriImageTraceMetrics,
                "audio": GeriAudioTraceMetrics,
            }
            cls = METRICS_CLS[self.modality]

            common = dict(
                trace_id=trace_id,
                e2e_latency_ms=latency_ms,
                queuing_delay_ms=queuing_delay_ms,
                input_seq_len=input_seq_len,
            )

            if self.modality == "image":
                metrics_list.append(cls(
                    **common,
                    height=int(tags.get("geri.engine.process_request.height", 0)),
                    width=int(tags.get("geri.engine.process_request.width", 0)),
                    num_inference_steps=int(tags.get("geri.engine.process_request.num_inference_steps", 0)),
                ))
            else:
                metrics_list.append(cls(**common))

        return metrics_list


class DisaggImageOtelProcessor(OtelProcessor):
    """Process OpenTelemetry traces for disaggregated image generation.

    Combines the LLM encoder span (``EngineCore.add_request``) and the
    Geri generator span (``geri.engine.process_request``) within the
    same trace.  Both queuing delays are subtracted so that
    ``e2e_latency_ms = llm_processing + geri_processing``.
    """

    def __init__(
        self,
        app_id: str,
        jaeger_host: str = "localhost",
        jaeger_port: int = 30686,
        service_name: str = "gateway",
    ):
        super().__init__(app_id, jaeger_host, jaeger_port, service_name)

    def _process_trace(self, trace: dict[str, Any]) -> list[DisaggImageTraceMetrics]:
        metrics_list: list[DisaggImageTraceMetrics] = []

        if "spans" not in trace:
            return metrics_list

        trace_id = trace.get("traceID", "unknown")

        # --- LLM encoder span (EngineCore.add_request) ---
        llm_queuing_us = 0
        llm_processing_us = 0
        for span in trace["spans"]:
            if "EngineCore.add_request" not in span.get("operationName", ""):
                continue
            span_duration_us = span.get("duration", 0)
            enqueue_time = self._extract_event_timestamp(span, "scheduler.enqueue")
            new_time = self._extract_event_timestamp(span, "scheduler.new")
            if enqueue_time and new_time:
                llm_queuing_us = new_time - enqueue_time
            llm_processing_us = span_duration_us - llm_queuing_us
            break  # Only one LLM encoder span per trace

        # --- Geri generator span (geri.engine.process_request) ---
        for span in trace["spans"]:
            if "geri.engine.process_request" not in span.get("operationName", ""):
                continue

            tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
            span_duration_us = span.get("duration", 0)

            enqueue_time = self._extract_event_timestamp(span, "geri.engine.scheduler.enqueue")
            schedule_time = self._extract_event_timestamp(span, "geri.engine.scheduler.schedule")

            if enqueue_time and schedule_time:
                geri_queuing_us = schedule_time - enqueue_time
                geri_processing_us = span_duration_us - geri_queuing_us
            else:
                geri_queuing_us = 0
                geri_processing_us = span_duration_us

            total_queuing_ms = (llm_queuing_us + geri_queuing_us) / 1000.0
            total_processing_ms = (llm_processing_us + geri_processing_us) / 1000.0

            input_seq_len = int(tags.get("geri.engine.process_request.dummy_seq_len", 0))

            metrics_list.append(DisaggImageTraceMetrics(
                trace_id=trace_id,
                e2e_latency_ms=total_processing_ms,
                queuing_delay_ms=total_queuing_ms,
                input_seq_len=input_seq_len,
                height=int(tags.get("geri.engine.process_request.height", 0)),
                width=int(tags.get("geri.engine.process_request.width", 0)),
                num_inference_steps=int(tags.get("geri.engine.process_request.num_inference_steps", 0)),
                llm_queuing_delay_ms=llm_queuing_us / 1000.0,
                llm_latency_ms=llm_processing_us / 1000.0,
                geri_queuing_delay_ms=geri_queuing_us / 1000.0,
                geri_latency_ms=geri_processing_us / 1000.0,
            ))

        return metrics_list


class LLMOtelProcessor(OtelProcessor):
    """Process OpenTelemetry traces for LLM generation requests.

    Extracts timing and token metrics from the Cornserve-integrated
    ``EngineCore.add_request`` span in vLLM, which tracks the full
    request lifecycle through the scheduler.
    """

    def _process_trace(self, trace: dict[str, Any]) -> list[LLMTraceMetrics]:
        """Process a single trace to extract LLM timing and token metrics."""
        metrics_list = []

        if "spans" not in trace:
            return metrics_list

        trace_id = trace.get("traceID", "unknown")

        for span in trace["spans"]:
            if "EngineCore.add_request" not in span.get("operationName", ""):
                continue

            span_duration_us = span.get("duration", 0)

            # Queuing delay = scheduler.new - scheduler.enqueue
            # (time waiting in the scheduler queue, excluding preprocessing).
            enqueue_time = self._extract_event_timestamp(span, "scheduler.enqueue")
            new_time = self._extract_event_timestamp(span, "scheduler.new")

            if enqueue_time and new_time:
                queuing_delay_us = new_time - enqueue_time
                queuing_delay_ms = queuing_delay_us / 1000.0
                latency_ms = (span_duration_us - queuing_delay_us) / 1000.0
            else:
                queuing_delay_ms = 0.0
                latency_ms = span_duration_us / 1000.0

            # Extract token counts and mm metadata from span attributes.
            tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
            num_prompt_tokens = int(tags.get("num_prompt_tokens", 0))
            num_output_tokens = int(tags.get("num_output_tokens", 0))

            # Extract per-item multimodal metadata (mm.{i}.modality, mm.{i}.encoder_tokens)
            num_mm_items = int(tags.get("mm.num_items", 0))
            mm_items: list[LLMMMItem] = []
            for i in range(num_mm_items):
                modality = str(tags.get(f"mm.{i}.modality", "unknown"))
                encoder_tokens = int(tags.get(f"mm.{i}.encoder_tokens", 0))
                mm_items.append(LLMMMItem(modality=modality, encoder_tokens=encoder_tokens))

            metrics_list.append(LLMTraceMetrics(
                trace_id=trace_id,
                e2e_latency_ms=latency_ms,
                queuing_delay_ms=queuing_delay_ms,
                num_prompt_tokens=num_prompt_tokens,
                num_output_tokens=num_output_tokens,
                mm_items=mm_items,
            ))

        return metrics_list


class MLLMOtelProcessor(OtelProcessor):
    """Process OpenTelemetry traces for MLLM (composite Eric + LLM) requests.

    Extracts timing from both Eric encoder spans
    (``Engine._request_receive_loop``) and LLM generation spans
    (``EngineCore.add_request``), combining their queuing delays so that
    ``queuing_delay_ms = eric_queuing + llm_queuing``.
    """

    def _process_trace(self, trace: dict[str, Any]) -> list[MLLMTraceMetrics]:
        """Process a single MLLM trace to extract combined Eric + LLM metrics."""
        metrics_list: list[MLLMTraceMetrics] = []

        if "spans" not in trace:
            return metrics_list

        trace_id = trace.get("traceID", "unknown")

        # --- Accumulate Eric encoder spans ---
        eric_total_queuing_us = 0
        eric_total_processing_us = 0
        for span in trace["spans"]:
            if "Engine._request_receive_loop" not in span.get("operationName", ""):
                continue
            span_duration_us = span.get("duration", 0)
            enqueue_time = self._extract_event_timestamp(span, "engine.scheduler.enqueue")
            dequeue_times = self._extract_all_event_timestamps(span, "engine.scheduler.dequeue")
            if enqueue_time and dequeue_times:
                queuing_us = min(dequeue_times) - enqueue_time
            else:
                queuing_us = 0
            eric_total_queuing_us += queuing_us
            eric_total_processing_us += (span_duration_us - queuing_us)

        # --- LLM generation span ---
        for span in trace["spans"]:
            if "EngineCore.add_request" not in span.get("operationName", ""):
                continue
            span_duration_us = span.get("duration", 0)
            enqueue_time = self._extract_event_timestamp(span, "scheduler.enqueue")
            new_time = self._extract_event_timestamp(span, "scheduler.new")

            if enqueue_time and new_time:
                llm_queuing_us = new_time - enqueue_time
            else:
                llm_queuing_us = 0
            llm_processing_us = span_duration_us - llm_queuing_us

            # Token + multimodal metadata
            tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
            num_prompt_tokens = int(tags.get("num_prompt_tokens", 0))
            num_output_tokens = int(tags.get("num_output_tokens", 0))

            num_mm_items = int(tags.get("mm.num_items", 0))
            mm_items: list[LLMMMItem] = []
            for i in range(num_mm_items):
                modality = str(tags.get(f"mm.{i}.modality", "unknown"))
                encoder_tokens = int(tags.get(f"mm.{i}.encoder_tokens", 0))
                mm_items.append(LLMMMItem(modality=modality, encoder_tokens=encoder_tokens))

            total_queuing_ms = (eric_total_queuing_us + llm_queuing_us) / 1000.0
            total_processing_ms = (eric_total_processing_us + llm_processing_us) / 1000.0

            metrics_list.append(MLLMTraceMetrics(
                trace_id=trace_id,
                e2e_latency_ms=total_processing_ms,
                queuing_delay_ms=total_queuing_ms,
                eric_queuing_delay_ms=eric_total_queuing_us / 1000.0,
                eric_latency_ms=eric_total_processing_us / 1000.0,
                llm_queuing_delay_ms=llm_queuing_us / 1000.0,
                llm_latency_ms=llm_processing_us / 1000.0,
                num_prompt_tokens=num_prompt_tokens,
                num_output_tokens=num_output_tokens,
                mm_items=mm_items,
            ))

        return metrics_list


class TalkerOtelProcessor(LLMOtelProcessor):
    """Process OpenTelemetry traces for Omni Talker embedding requests.

    The Talker is vLLM-backed and shares the same `EngineCore.add_request`
    span structure as the standard LLM processor.
    """


class TalkerVocoderOtelProcessor(LLMOtelProcessor):
    """Process OpenTelemetry traces for Omni Talker Vocoder requests.

    The Talker Vocoder is vLLM-backed and shares the same `EngineCore.add_request`
    span structure as the standard LLM processor.
    """


class OmniRouterOtelProcessor(OtelProcessor):
    """Process OpenTelemetry traces for OmniRouter requests.

    The OmniRouter pipeline is: Eric encoders → Thinker (vLLM) → Talker (vLLM).
    Non-audio requests skip the Talker stage.

    A trace may contain:
      - Zero or more ``Engine._request_receive_loop`` spans (Eric encoders)
      - One ``EngineCore.add_request`` span for the Thinker
      - Optionally one ``EngineCore.add_request`` span for the Talker

    The Thinker always starts before the Talker (it produces text that the
    Talker consumes), so the two LLM spans are distinguished by ``startTime``.

    Returns exactly **one** ``OmniRouterTraceMetrics`` per trace.
    ``e2e_latency_ms`` and ``llm_latency_ms`` report the thinker only.
    The talker is reported in dedicated ``talker_*`` fields.
    """

    def _process_trace(self, trace: dict[str, Any]) -> list[OmniRouterTraceMetrics]:
        metrics_list: list[OmniRouterTraceMetrics] = []

        if "spans" not in trace:
            return metrics_list

        trace_id = trace.get("traceID", "unknown")

        # --- Eric encoder spans (run in parallel / fork-join) ---
        # Queuing delay = max across encoders (slowest determines when
        # downstream can start).  Processing time = max (parallel execution).
        eric_queuing_delays_us: list[int] = []
        eric_processing_times_us: list[int] = []
        for span in trace["spans"]:
            if "Engine._request_receive_loop" not in span.get("operationName", ""):
                continue
            span_duration_us = span.get("duration", 0)
            enqueue_time = self._extract_event_timestamp(span, "engine.scheduler.enqueue")
            dequeue_times = self._extract_all_event_timestamps(span, "engine.scheduler.dequeue")
            if enqueue_time and dequeue_times:
                queuing_us = min(dequeue_times) - enqueue_time
            else:
                queuing_us = 0
            eric_queuing_delays_us.append(queuing_us)
            eric_processing_times_us.append(span_duration_us - queuing_us)
        eric_max_queuing_us = max(eric_queuing_delays_us) if eric_queuing_delays_us else 0
        eric_max_processing_us = max(eric_processing_times_us) if eric_processing_times_us else 0

        # --- Collect all EngineCore.add_request spans, sorted by startTime ---
        llm_spans = sorted(
            (s for s in trace["spans"]
             if "EngineCore.add_request" in s.get("operationName", "")),
            key=lambda s: s.get("startTime", 0),
        )
        if not llm_spans:
            return metrics_list

        # First span = Thinker (always starts first in the pipeline)
        thinker_span = llm_spans[0]
        # Second span (if present) = Talker (audio generation)
        talker_span = llm_spans[1] if len(llm_spans) >= 2 else None

        # --- Thinker metrics ---
        def _extract_llm_timing(span: dict) -> tuple[int, int]:
            span_duration_us = span.get("duration", 0)
            enqueue_time = self._extract_event_timestamp(span, "scheduler.enqueue")
            new_time = self._extract_event_timestamp(span, "scheduler.new")
            if enqueue_time and new_time:
                queuing_us = new_time - enqueue_time
            else:
                queuing_us = 0
            processing_us = span_duration_us - queuing_us
            return queuing_us, processing_us

        thinker_queuing_us, thinker_processing_us = _extract_llm_timing(thinker_span)

        thinker_tags = {tag["key"]: tag["value"] for tag in thinker_span.get("tags", [])}
        num_prompt_tokens = int(thinker_tags.get("num_prompt_tokens", 0))
        num_output_tokens = int(thinker_tags.get("num_output_tokens", 0))

        num_mm_items = int(thinker_tags.get("mm.num_items", 0))
        mm_items: list[LLMMMItem] = []
        for i in range(num_mm_items):
            modality = str(thinker_tags.get(f"mm.{i}.modality", "unknown"))
            encoder_tokens = int(thinker_tags.get(f"mm.{i}.encoder_tokens", 0))
            mm_items.append(LLMMMItem(modality=modality, encoder_tokens=encoder_tokens))

        # --- Talker metrics (if audio) ---
        talker_queuing_us = 0
        talker_processing_us = 0
        talker_output_tokens = 0
        return_audio = talker_span is not None
        if talker_span is not None:
            talker_queuing_us, talker_processing_us = _extract_llm_timing(talker_span)
            talker_tags = {tag["key"]: tag["value"] for tag in talker_span.get("tags", [])}
            talker_output_tokens = int(talker_tags.get("num_output_tokens", 0))

        # --- Combined metrics ---
        # queuing_delay = max(eric_queuing) + thinker_queuing + talker_queuing
        # e2e_latency   = max(eric_processing) + thinker_processing + talker_processing
        total_queuing_ms = (eric_max_queuing_us + thinker_queuing_us + talker_queuing_us) / 1000.0
        total_processing_ms = (eric_max_processing_us + thinker_processing_us + talker_processing_us) / 1000.0

        metrics_list.append(OmniRouterTraceMetrics(
            trace_id=trace_id,
            e2e_latency_ms=total_processing_ms,
            queuing_delay_ms=total_queuing_ms,
            eric_queuing_delay_ms=eric_max_queuing_us / 1000.0,
            eric_latency_ms=eric_max_processing_us / 1000.0,
            llm_queuing_delay_ms=thinker_queuing_us / 1000.0,
            llm_latency_ms=thinker_processing_us / 1000.0,
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            mm_items=mm_items,
            talker_queuing_delay_ms=talker_queuing_us / 1000.0,
            talker_latency_ms=talker_processing_us / 1000.0,
            talker_output_tokens=talker_output_tokens,
            return_audio=return_audio,
        ))

        return metrics_list


class DisaggregatedMLLMTraceMetricsConfig(BaseModel):
    """Shared prefill/decode metric fields for PD and EPD MLLM trace metrics."""

    prefill_queuing_delay_ms: float
    prefill_latency_ms: float
    prefill_kv_transfer_latency_ms: float
    decode_queuing_delay_ms: float
    decode_latency_ms: float
    num_prompt_tokens: int
    num_output_tokens: int
    mm_items: list[LLMMMItem]


class PDMLLMTraceMetrics(TraceMetrics, DisaggregatedMLLMTraceMetricsConfig):
    """Processed metrics from a PD (disaggregated prefill+decode) MLLM trace.

    A PD trace contains two ``EngineCore.add_request`` spans: one from the
    prefill worker and one from the decode worker. They are separated by
    ``startTime`` (prefill always starts first).

    ``e2e_latency_ms``   = ``prefill_latency_ms + decode_latency_ms``
    ``queuing_delay_ms`` = ``prefill_queuing_ms + decode_queuing_ms``
    """


class EPDMLLMTraceMetrics(TraceMetrics, DisaggregatedMLLMTraceMetricsConfig):
    """Processed metrics from an EPD (encoder+prefill+decode) MLLM trace.

    ``e2e_latency_ms``   = ``eric_latency_ms + prefill_latency_ms + decode_latency_ms``
    ``queuing_delay_ms`` = ``eric_queuing_ms + prefill_queuing_ms + decode_queuing_ms``
    """
    eric_queuing_delay_ms: float
    eric_latency_ms: float


def _extract_llm_span_timing(
    span: dict[str, Any],
    *,
    is_decode: bool = False,
) -> tuple[int, int, int]:
    """Extract (queuing_us, processing_us, kv_transfer_us) from an EngineCore.add_request span.

    ``processing_us`` is measured from span start to the ``scheduler.first_token``
    event when present (prefill compute only, before KV-cache transfer wait).
    Falls back to ``span_duration_us - queuing_us`` when the event is absent
    (decode spans, or traces recorded before the event was added).

    ``kv_transfer_us`` is measured from ``scheduler.first_token`` to span end
    (the time the prefill worker is blocked waiting for the decode worker to pull
    the KV cache).  Zero when the event is absent.

    When ``is_decode`` is True the ``scheduler.first_token`` event is ignored
    because on decode spans it fires *after* the long scheduler queue wait,
    so using it would misattribute queuing time as processing.
    """
    span_start_us = span.get("startTime", 0)
    span_duration_us = span.get("duration", 0)
    enqueue_time_us = None
    new_time_us = None
    first_token_time_us = None
    for log in span.get("logs", []):
        for field in log.get("fields", []):
            if field.get("key") == "event":
                val = field.get("value")
                if val == "scheduler.enqueue":
                    enqueue_time_us = log.get("timestamp")
                elif val == "scheduler.new":
                    new_time_us = log.get("timestamp")
                elif val == "scheduler.first_token":
                    first_token_time_us = log.get("timestamp")
    if enqueue_time_us and new_time_us:
        queuing_us = new_time_us - enqueue_time_us
    else:
        queuing_us = 0
    if not is_decode and first_token_time_us and span_start_us:
        processing_us = first_token_time_us - span_start_us
        kv_transfer_us = (span_start_us + span_duration_us) - first_token_time_us
    else:
        processing_us = span_duration_us - queuing_us
        kv_transfer_us = 0
    return queuing_us, processing_us, kv_transfer_us


class PDMLLMOtelProcessor(OtelProcessor):
    """Process OpenTelemetry traces for PD (disaggregated prefill+decode) MLLM requests.

    A PD trace contains exactly two ``EngineCore.add_request`` spans: one from
    the prefill worker and one from the decode worker.  Since
    ``DisaggregatedMLLMTask.invoke()`` calls prefill before decode, the spans
    are distinguished by ``startTime`` — the earlier span is prefill.

    Returns exactly **one** ``PDMLLMTraceMetrics`` per trace.
    """

    def _process_trace(self, trace: dict[str, Any]) -> list[PDMLLMTraceMetrics]:
        if "spans" not in trace:
            return []

        trace_id = trace.get("traceID", "unknown")

        llm_spans = sorted(
            (s for s in trace["spans"]
             if "EngineCore.add_request" in s.get("operationName", "")),
            key=lambda s: s.get("startTime", 0),
        )
        if len(llm_spans) < 2:
            return []

        prefill_span, decode_span = llm_spans[0], llm_spans[1]

        prefill_queuing_us, prefill_processing_us, prefill_kv_transfer_us = _extract_llm_span_timing(prefill_span)
        decode_queuing_us, decode_processing_us, _ = _extract_llm_span_timing(decode_span)

        # Token metadata comes from the prefill span (it sees the full prompt).
        prefill_tags = {tag["key"]: tag["value"] for tag in prefill_span.get("tags", [])}
        num_prompt_tokens = int(prefill_tags.get("num_prompt_tokens", 0))
        num_output_tokens = int(prefill_tags.get("num_output_tokens", 0))
        num_mm_items = int(prefill_tags.get("mm.num_items", 0))
        mm_items: list[LLMMMItem] = []
        for i in range(num_mm_items):
            modality = str(prefill_tags.get(f"mm.{i}.modality", "unknown"))
            encoder_tokens = int(prefill_tags.get(f"mm.{i}.encoder_tokens", 0))
            mm_items.append(LLMMMItem(modality=modality, encoder_tokens=encoder_tokens))

        total_queuing_ms = (prefill_queuing_us + decode_queuing_us) / 1000.0
        total_processing_ms = (prefill_processing_us + decode_processing_us) / 1000.0

        return [PDMLLMTraceMetrics(
            trace_id=trace_id,
            e2e_latency_ms=total_processing_ms,
            queuing_delay_ms=total_queuing_ms,
            prefill_queuing_delay_ms=prefill_queuing_us / 1000.0,
            prefill_latency_ms=prefill_processing_us / 1000.0,
            prefill_kv_transfer_latency_ms=prefill_kv_transfer_us / 1000.0,
            decode_queuing_delay_ms=decode_queuing_us / 1000.0,
            decode_latency_ms=decode_processing_us / 1000.0,
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            mm_items=mm_items,
        )]


class EPDMLLMOtelProcessor(OtelProcessor):
    """Process OpenTelemetry traces for EPD (encoder+prefill+decode) MLLM requests.

    An EPD trace contains:
      - Zero or more ``Engine._request_receive_loop`` spans (Eric encoders, parallel)
      - Two ``EngineCore.add_request`` spans (prefill and decode, sequential)

    Eric encoding is parallel, so its contribution is ``max()`` across encoder
    spans (same policy as :class:`OmniRouterOtelProcessor`).

    Returns exactly **one** ``EPDMLLMTraceMetrics`` per trace.
    """

    def _process_trace(self, trace: dict[str, Any]) -> list[EPDMLLMTraceMetrics]:
        if "spans" not in trace:
            return []

        trace_id = trace.get("traceID", "unknown")

        # --- Eric encoder spans (parallel) ---
        eric_queuing_delays_us: list[int] = []
        eric_processing_times_us: list[int] = []
        for span in trace["spans"]:
            if "Engine._request_receive_loop" not in span.get("operationName", ""):
                continue
            span_duration_us = span.get("duration", 0)
            enqueue_time = self._extract_event_timestamp(span, "engine.scheduler.enqueue")
            dequeue_times = self._extract_all_event_timestamps(span, "engine.scheduler.dequeue")
            if enqueue_time and dequeue_times:
                queuing_us = min(dequeue_times) - enqueue_time
            else:
                queuing_us = 0
            eric_queuing_delays_us.append(queuing_us)
            eric_processing_times_us.append(span_duration_us - queuing_us)

        eric_queuing_us = max(eric_queuing_delays_us) if eric_queuing_delays_us else 0
        eric_processing_us = max(eric_processing_times_us) if eric_processing_times_us else 0

        # --- Prefill + decode spans (sorted by startTime) ---
        llm_spans = sorted(
            (s for s in trace["spans"]
             if "EngineCore.add_request" in s.get("operationName", "")),
            key=lambda s: s.get("startTime", 0),
        )
        if len(llm_spans) < 2:
            return []

        prefill_span, decode_span = llm_spans[0], llm_spans[1]

        prefill_queuing_us, prefill_processing_us, prefill_kv_transfer_us = _extract_llm_span_timing(prefill_span)
        decode_queuing_us, decode_processing_us, _ = _extract_llm_span_timing(decode_span, is_decode=True)

        prefill_tags = {tag["key"]: tag["value"] for tag in prefill_span.get("tags", [])}
        decode_tags = {tag["key"]: tag["value"] for tag in decode_span.get("tags", [])}
        num_prompt_tokens = int(prefill_tags.get("num_prompt_tokens", 0))
        num_output_tokens = int(decode_tags.get("num_output_tokens", 0))
        num_mm_items = int(prefill_tags.get("mm.num_items", 0))
        mm_items: list[LLMMMItem] = []
        for i in range(num_mm_items):
            modality = str(prefill_tags.get(f"mm.{i}.modality", "unknown"))
            encoder_tokens = int(prefill_tags.get(f"mm.{i}.encoder_tokens", 0))
            mm_items.append(LLMMMItem(modality=modality, encoder_tokens=encoder_tokens))

        total_queuing_ms = (eric_queuing_us + prefill_queuing_us + decode_queuing_us) / 1000.0
        total_processing_ms = (eric_processing_us + prefill_processing_us + decode_processing_us) / 1000.0

        return [EPDMLLMTraceMetrics(
            trace_id=trace_id,
            e2e_latency_ms=total_processing_ms,
            queuing_delay_ms=total_queuing_ms,
            eric_queuing_delay_ms=eric_queuing_us / 1000.0,
            eric_latency_ms=eric_processing_us / 1000.0,
            prefill_queuing_delay_ms=prefill_queuing_us / 1000.0,
            prefill_latency_ms=prefill_processing_us / 1000.0,
            prefill_kv_transfer_latency_ms=prefill_kv_transfer_us / 1000.0,
            decode_queuing_delay_ms=decode_queuing_us / 1000.0,
            decode_latency_ms=decode_processing_us / 1000.0,
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            mm_items=mm_items,
        )]


if __name__ == "__main__":
    # # --- Previous driver: fetch from Jaeger and process ---
    # # Process image traces
    # print("Processing image traces...")
    # image_processor = EricOtelProcessor(
    #     app_id="app-6947977f704b4a97880c553d6ed4591a",
    #     modality="image",
    #     jaeger_host="localhost",
    #     jaeger_port=30686,
    #     service_name="gateway",
    # )
    # image_output_dir = Path("data/eric/OpenGVLab/InternVL3-38B/image/traces")
    # image_summary = image_processor.process_and_save(
    #     output_dir=image_output_dir,
    #     limit=1000,
    #     lookback="1h",
    # )
    # print(f"Image: {image_summary.total_requests} requests, "
    #       f"avg latency: {image_summary.avg_latency_ms:.2f}ms")
    #
    # # Process video traces
    # print("\nProcessing video traces...")
    # video_processor = EricOtelProcessor(
    #     app_id="app-bb3ab03fba614741a75cee3eeef8b525",
    #     modality="video",
    #     jaeger_host="localhost",
    #     jaeger_port=30686,
    #     service_name="gateway",
    # )
    # video_output_dir = Path("data/eric/OpenGVLab/InternVL3-38B/video/traces")
    # video_summary = video_processor.process_and_save(
    #     output_dir=video_output_dir,
    #     limit=1000,
    #     lookback="1h",
    # )
    # print(f"Video: {video_summary.total_requests} requests, "
    #       f"avg latency: {video_summary.avg_latency_ms:.2f}ms")
    #
    # # Process audio traces
    # print("\nProcessing audio traces...")
    # audio_processor = EricOtelProcessor(
    #     app_id="app-a5b925b601a14277b2779d46c44f70a8",
    #     modality="audio",
    #     jaeger_host="localhost",
    #     jaeger_port=30686,
    #     service_name="gateway",
    # )
    # audio_output_dir = Path("data/eric/Qwen/Qwen2.5-Omni-3B/audio/traces")
    # audio_summary = audio_processor.process_and_save(
    #     output_dir=audio_output_dir,
    #     limit=1000,
    #     lookback="1h",
    # )
    # print(f"Audio: {audio_summary.total_requests} requests, "
    #       f"avg latency: {audio_summary.avg_latency_ms:.2f}ms")
    #
    # print("\n" + "="*60)
    # print("SUMMARY - ALL MODALITIES")
    # print("="*60)
    #
    # for name, summary in [("Image", image_summary), ("Video", video_summary), ("Audio", audio_summary)]:
    #     print(f"\n{name.upper()}")
    #     print(f"  Total traces: {summary.total_traces}")
    #     print(f"  Total requests: {summary.total_requests}")
    #     print(f"  Latency (queuing removed): {summary.avg_latency_ms:.2f} ms (median: {summary.median_latency_ms:.2f} ms)")
    #     print(f"    P90: {summary.p90_latency_ms:.2f} ms, P95: {summary.p95_latency_ms:.2f} ms, P99: {summary.p99_latency_ms:.2f} ms")
    #
    # print("="*60)

    # # --- Reprocess saved Eric raw traces ---
    # otel_dir = Path("data/eric/Qwen/Qwen3-Omni-30B-A3B-Instruct/video+replicas1+tp1+maxbs8/otel")
    #
    # processor = EricOtelProcessor(
    #     app_id="app-2d78c88e0b9c4cc0a2180b5579e78099",
    #     modality="video",
    # )
    #
    # print(f"Loading raw traces from {otel_dir / 'raw'}...")
    # summary = processor.load_and_process(otel_dir)
    #
    # print(f"\nTotal traces:   {summary.total_traces}")
    # print(f"Total requests: {summary.total_requests}")
    # print(f"Avg latency:    {summary.avg_latency_ms:.2f} ms (median: {summary.median_latency_ms:.2f} ms)")
    # print(f"  P90: {summary.p90_latency_ms:.2f} ms, P95: {summary.p95_latency_ms:.2f} ms, P99: {summary.p99_latency_ms:.2f} ms")
    # print(f"Avg queuing:    {summary.avg_queuing_delay_ms:.2f} ms ({summary.queuing_delay_percentage:.1f}%)")
    # print(f"Avg e2e:        {summary.avg_e2e_latency_ms:.2f} ms")

    # # --- Reprocess saved LLM raw traces ---
    # otel_dir = Path("data/llm/Qwen/Qwen2.5-VL-7B-Instruct/replicas1+tp2+bs32+gpu0.9/otel")
    #
    # processor = LLMOtelProcessor(
    #     app_id="app-37dac24a1649407a9099dae3b18296e3",
    # )
    #
    # print(f"Loading raw traces from {otel_dir / 'raw'}...")
    # summary = processor.load_and_process(otel_dir)
    #
    # print(f"\nTotal traces:   {summary.total_traces}")
    # print(f"Total requests: {summary.total_requests}")
    # print(f"Avg latency:    {summary.avg_latency_ms:.2f} ms (median: {summary.median_latency_ms:.2f} ms)")
    # print(f"  P90: {summary.p90_latency_ms:.2f} ms, P95: {summary.p95_latency_ms:.2f} ms, P99: {summary.p99_latency_ms:.2f} ms")
    # print(f"Avg queuing:    {summary.avg_queuing_delay_ms:.2f} ms ({summary.queuing_delay_percentage:.1f}%)")
    # print(f"Avg e2e:        {summary.avg_e2e_latency_ms:.2f} ms")

    # # --- Fetch and process Geri image traces ---
    # import asyncio
    #
    # processor = GeriOtelProcessor(
    #     app_id="app-a62b324f75a645aaa00ec538d6354f55",
    #     modality="image",
    # )
    #
    # print("Fetching Geri image traces from Jaeger...")
    # traces = asyncio.run(processor._get_traces(limit=1000, lookback="1h"))
    # print(f"Found {len(traces)} traces")
    #
    # for trace in traces:
    #     metrics = processor._process_trace(trace)
    #     for m in metrics:
    #         line = (f"  trace={m.trace_id[:12]}  latency={m.e2e_latency_ms:.1f}ms  "
    #                 f"queuing={m.queuing_delay_ms:.1f}ms  input_seq_len={m.input_seq_len}")
    #         if isinstance(m, GeriImageTraceMetrics):
    #             line += f"  height={m.height}  width={m.width}  steps={m.num_inference_steps}"
    #         print(line)
    #
    # output_dir = Path("data/geri/image/test_otel")
    # summary = asyncio.run(processor.process_and_save(output_dir))
    #
    # print(f"\nTotal traces:   {summary.total_traces}")
    # print(f"Total requests: {summary.total_requests}")
    # print(f"Avg latency:    {summary.avg_latency_ms:.2f} ms (median: {summary.median_latency_ms:.2f} ms)")
    # print(f"  P90: {summary.p90_latency_ms:.2f} ms, P95: {summary.p95_latency_ms:.2f} ms, P99: {summary.p99_latency_ms:.2f} ms")
    # print(f"Avg queuing:    {summary.avg_queuing_delay_ms:.2f} ms ({summary.queuing_delay_percentage:.1f}%)")
    # print(f"Avg e2e:        {summary.avg_e2e_latency_ms:.2f} ms")

    # --- Reprocess saved MLLM (ModServe) raw traces ---
    otel_dir = Path(
        "data/modserve/Qwen/Qwen2.5-VL-7B-Instruct/"
        "erics2+llms2+eric_bs2+llm_tp1+llm_bs32+llm_gpu0.9/"
        "A100+controlled_image_chat+requests300+input1000+output300"
        "+img_w1920_h1080+count1+prob1.0+seed48105+rate5.0/otel"
    )

    processor = MLLMOtelProcessor(
        app_id="app-36597a0e6bb8418c9c1820d0c0bfd99e",
    )

    print(f"Loading raw MLLM traces from {otel_dir / 'raw.json'}...")
    summary = processor.load_and_process(otel_dir)

    print(f"\nTotal traces:   {summary.total_traces}")
    print(f"Total requests: {summary.total_requests}")
    print(f"Avg processing: {summary.avg_latency_ms:.2f} ms (median: {summary.median_latency_ms:.2f} ms)")
    print(f"  P90: {summary.p90_latency_ms:.2f} ms, P95: {summary.p95_latency_ms:.2f} ms, P99: {summary.p99_latency_ms:.2f} ms")
    print(f"Avg queuing:    {summary.avg_queuing_delay_ms:.2f} ms ({summary.queuing_delay_percentage:.1f}%)")
    print(f"Avg e2e:        {summary.avg_e2e_latency_ms:.2f} ms")

    # Print per-trace breakdown for first 5 traces
    with open(otel_dir / "raw.json") as f:
        import json as _json
        traces = _json.load(f).get("data", [])

    print("\nPer-trace breakdown (first 10):")
    for trace in traces[:10]:
        for m in processor._process_trace(trace):
            print(
                f"  trace={m.trace_id[:12]}  "
                f"eric_q={m.eric_queuing_delay_ms:.0f}ms  eric_proc={m.eric_latency_ms:.0f}ms  "
                f"llm_q={m.llm_queuing_delay_ms:.0f}ms  llm_proc={m.llm_latency_ms:.0f}ms  "
                f"total_q={m.queuing_delay_ms:.0f}ms  total_proc={m.e2e_latency_ms:.0f}ms  "
                f"tokens={m.num_prompt_tokens}/{m.num_output_tokens}  "
                f"mm={[(item.modality, item.encoder_tokens) for item in m.mm_items]}"
            )
