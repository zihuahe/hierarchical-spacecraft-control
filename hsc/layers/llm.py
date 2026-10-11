"""Thin wrapper around the Anthropic SDK for JSON-schema structured calls with latency timing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

OPUS_MODEL = "claude-opus-5-5"
HAIKU_MODEL = "claude-haiku-5-5"

# Models that accept the server-side refusal fallback ("default" routes by refusal category).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


@dataclass
class CallResult:
    data: dict
    latency_s: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    extra: dict = field(default_factory=dict)


class StructuredLLM:
    def __init__(self, model: str, effort: str = "low", max_tokens: int = 4000, client=None):
        import anthropic  # imported lazily so mock-only runs don't need the SDK

        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()

    def call(self, system: str, content, schema: dict) -> CallResult:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        t0 = time.perf_counter()
        if self.model in _FALLBACK_MODELS:
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - t0
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"{self.model} refused: {getattr(resp, 'stop_details', None)}")
        text = next(b.text for b in resp.content if b.type == "text")
        return CallResult(
            data=json.loads(text),
            latency_s=latency,
            model=getattr(resp, "model", self.model),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
