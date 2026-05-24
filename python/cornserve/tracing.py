"""OpenTelemetry configuration for the cornserve services."""

from __future__ import annotations

import os
from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from cornserve.constants import K8S_OTEL_GRPC_URL


class _FilteringExporter(SpanExporter):
    """A SpanExporter wrapper that drops noisy, low-value spans.

    Filters out:
    - K8s API watch long-polling calls from kubernetes_asyncio CRD watchers.
    - Periodic gRPC health check calls (CheckHealth).
    """

    def __init__(self, delegate: SpanExporter, k8s_host: str | None) -> None:
        self._delegate = delegate
        self._k8s_host = k8s_host

    def _should_exclude(self, span: ReadableSpan) -> bool:
        attrs = span.attributes or {}

        # Exclude gRPC health check calls.
        rpc_method = str(attrs.get("rpc.method", ""))
        if rpc_method == "CheckHealth":
            return True

        # Exclude K8s API watch calls (background long-polls).
        if self._k8s_host:
            url = str(attrs.get("http.url", "") or attrs.get("url.full", ""))
            if self._k8s_host in url and ("watch=True" in url or "watch=true" in url):
                return True

        return False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        filtered = [s for s in spans if not self._should_exclude(s)]
        if not filtered:
            return SpanExportResult.SUCCESS
        return self._delegate.export(filtered)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def configure_otel(name: str) -> None:
    """Configure OpenTelemetry for the given service name."""
    resource = Resource.create({"service.name": name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=K8S_OTEL_GRPC_URL,
        timeout=10,
    )

    # Filter out noisy spans: K8s API watch long-polls and gRPC health checks.
    k8s_host = os.environ.get("KUBERNETES_SERVICE_HOST")
    exporter = _FilteringExporter(exporter, k8s_host)
    processor = BatchSpanProcessor(
        exporter,
        max_queue_size=8192,
        schedule_delay_millis=2000,
        max_export_batch_size=512,
        export_timeout_millis=10000,
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
