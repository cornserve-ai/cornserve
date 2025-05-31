# Tracing Developer guide

We employ OpenTelemetry for observability. Below are some of the conventions we use.

Generally, we use auto-instrumentation provided by OpenTelemetry, e.g., FastAPI, gRPC, HTTPX.

## Spans

Usually named with `ClassName.function_name`.

## Attributes

Usually named with `namespace.subroutine.attribute_name`.
`namespace` is typically the name of the service, like `gateway`.

## Events

Usually named with `action.event_name`.
Use spans for things that happen over time (e.g., a subroutine), where tracking the start and end is important.
On the other hand, use events for singular occurrences that happen at a specific moment in time.

## Test
When testing locally, you can disable OTEL tracing through `OTEL_SDK_DISABLED=true`.
