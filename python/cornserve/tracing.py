"""OpenTelemetry configuration for the cornserve services."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

from cornserve.constants import OTEL_COLLECTOR_URL


def configure_otel(name: str) -> None:
    """Configure OpenTelemetry for the given service name."""
    resource = Resource.create({"service.name": name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=OTEL_COLLECTOR_URL)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
