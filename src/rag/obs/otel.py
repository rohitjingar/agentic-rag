"""OpenTelemetry tracing — one span per pipeline stage, exported to Jaeger.

Tracing is opt-in (otel_enabled): when off, `span()` is a zero-overhead no-op so
tests and the eval runner never need a collector. When on, each stage
(cache/retrieve/generate) becomes a span carrying token + cost + latency
attributes, so a per-query trace in Jaeger shows exactly where time and
shadow-dollars went — the same OTel wiring pattern as agent-gateway.
"""

from __future__ import annotations

from contextlib import contextmanager

_tracer = None


def setup_tracing(service_name: str, endpoint: str) -> None:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    global _tracer
    _tracer = trace.get_tracer("agentic-rag")


@contextmanager
def span(name: str, **attributes):
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield current


def set_attrs(current, **attributes) -> None:
    if current is not None:
        for key, value in attributes.items():
            current.set_attribute(key, value)
