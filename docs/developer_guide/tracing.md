# Tracing Developer guide

We employ opentelemetry to achieve observability.

## Spans
Named after `ClassName.function_name`

## Attributes
Named after `namespace.function.<some attribute>`

## Events
Named after `<some action>.status`
There are four types of events that are currently in interest for analysis
- Within single span: action.start -> action.done
    * Memory back-pressure
    * `Gloo` latency
    * Scheduling delay
- Within single trace, across spans: action.start -> action.done
    * D2H copy in the `SidecarSender` client.
    - this requires the name of the action must be unique within some parent span
- Within single span, action.start -> action.resume* -> action.stop
    * vLLM scheduler
- Using event to store attributes to avoid being overwritten
    * Eric scheduler: multiple data item of a request could be scheduled in
    different batches
