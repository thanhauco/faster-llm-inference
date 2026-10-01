"""Load *your own* traffic for replay.

"Measure on your own traffic before trusting any speedup number." Export a
sample of real requests (prompts, chat messages, max_tokens, temperature) to
JSONL and replay it against a baseline server and a speculative one.

Accepted record shapes, one JSON object per line::

    {"prompt": "...", "max_tokens": 256, "temperature": 0.0}
    {"messages": [{"role": "user", "content": "..."}], "max_tokens": 512}
    {"body": {...OpenAI request body...}}          # e.g. from a gateway log

Unknown keys inside an OpenAI body (``top_p``, ``stop``, ...) are forwarded.
"""

from __future__ import annotations

import json
import random

from specdec.serving.client import Request

_KNOWN = {"prompt", "messages", "max_tokens", "max_completion_tokens", "temperature", "model", "stream",
          "stream_options"}


def record_to_request(record: dict, default_max_tokens: int = 256, override_max_tokens: int | None = None,
                      override_temperature: float | None = None) -> Request:
    body = record.get("body", record)
    if "prompt" not in body and "messages" not in body:
        raise ValueError("record needs 'prompt' or 'messages'")
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens
    temperature = body.get("temperature", 0.0)
    extra = {k: v for k, v in body.items() if k not in _KNOWN}
    return Request(
        prompt=body.get("prompt"),
        messages=body.get("messages"),
        max_tokens=int(override_max_tokens or max_tokens),
        temperature=float(temperature if override_temperature is None else override_temperature),
        extra=extra,
    )


def load_jsonl(path: str, limit: int | None = None, shuffle: bool = False, seed: int = 0,
               **kwargs) -> list[Request]:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if shuffle:
        random.Random(seed).shuffle(records)
    if limit is not None:
        records = records[:limit]
    return [record_to_request(r, **kwargs) for r in records]
