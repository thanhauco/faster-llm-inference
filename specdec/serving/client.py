"""Streaming client for OpenAI-compatible servers (vLLM, SGLang, TGI, ...).

Measures what users feel: time to first token (TTFT), time per output token
(TPOT) and end-to-end latency, per request.

Note on speculative decoding and streaming: one SSE chunk can carry several
tokens (everything accepted in a verification step), so "inter-chunk latency"
is not "inter-token latency". We therefore report TPOT computed from the token
count in ``usage`` (requested with ``stream_options.include_usage``) rather
than from chunk gaps.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from specdec.serving.http import connect, get_json


@dataclass
class Request:
    prompt: str | None = None
    messages: list[dict] | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_chat(self) -> bool:
        return self.messages is not None

    def payload(self, model: str) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.is_chat:
            body["messages"] = self.messages
        else:
            body["prompt"] = self.prompt
        body.update(self.extra)
        return body


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float = 0.0
    latency_s: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    num_chunks: int = 0
    error: str = ""
    text: str = ""

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first."""
        if self.output_tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)

    @property
    def tokens_per_chunk(self) -> float | None:
        """>1 hints that the server streams several accepted tokens per step."""
        return self.output_tokens / self.num_chunks if self.num_chunks else None


def list_models(base_url: str, api_key: str | None = None, timeout: float = 10.0) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = get_json(base_url.rstrip("/") + "/models", timeout, headers)
    return [m["id"] for m in data.get("data", [])]


def _chunk_text(chunk: dict) -> str:
    out = []
    for choice in chunk.get("choices") or []:
        if "delta" in choice:
            out.append(choice["delta"].get("content") or "")
        else:
            out.append(choice.get("text") or "")
    return "".join(out)


def send(base_url: str, model: str, req: Request, api_key: str | None = None, timeout: float = 600.0,
         keep_text: bool = False) -> RequestResult:
    """Send one streaming request and time it."""
    endpoint = "/chat/completions" if req.is_chat else "/completions"
    url = base_url.rstrip("/") + endpoint
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(req.payload(model)).encode()

    start = time.perf_counter()
    first = None
    chunks = 0
    usage_tokens = None
    prompt_tokens = 0
    text_parts: list[str] = []
    conn, path = connect(url, timeout)
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status >= 400:
            msg = resp.read().decode("utf-8", "replace")[:300]
            return RequestResult(False, error=f"HTTP {resp.status}: {msg}", latency_s=time.perf_counter() - start)
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens", usage_tokens)
                prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
            piece = _chunk_text(chunk)
            if piece:
                if first is None:
                    first = time.perf_counter()
                chunks += 1
                if keep_text:
                    text_parts.append(piece)
    except Exception as exc:  # network errors, malformed JSON, timeouts
        return RequestResult(False, error=f"{type(exc).__name__}: {exc}", latency_s=time.perf_counter() - start)
    finally:
        conn.close()
    end = time.perf_counter()
    if first is None:
        return RequestResult(False, error="no tokens received", latency_s=end - start)
    return RequestResult(
        ok=True,
        ttft_s=first - start,
        latency_s=end - start,
        output_tokens=usage_tokens if usage_tokens is not None else chunks,
        prompt_tokens=prompt_tokens,
        num_chunks=chunks,
        text="".join(text_parts),
    )
