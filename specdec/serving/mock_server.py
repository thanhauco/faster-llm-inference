"""A tiny OpenAI-compatible server backed by the toy model, for testing the tooling end to end.

It speaks just enough of vLLM's API for ``specdec replay`` and ``specdec
metrics`` to work without a GPU:

* ``GET  /v1/models``
* ``POST /v1/completions`` and ``/v1/chat/completions`` (streaming and not)
* ``GET  /metrics`` with vLLM-style ``vllm:spec_decode_*`` counters

Text is mapped to the toy vocabulary character by character, so outputs are
gibberish by design. Each engine step sleeps for the time the roofline cost
model predicts at the current number of in-flight requests, so speculation
produces realistic-looking latency differences in the replay report.

    specdec mock-server --port 8000                      # baseline
    specdec mock-server --port 8001 --method ngram -k 5  # n-gram speculation
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from specdec.cost_model import CostModel
from specdec.drafters.base import Drafter
from specdec.engine import StepEvent, speculative_stream
from specdec.models import ToyLM
from specdec.sampling import apply_temperature, sample

PRINTABLE = 95  # ASCII 32..126


def encode(text: str, vocab_size: int) -> list[int]:
    toks = [ord(c) % vocab_size for c in text]
    return toks or [1]


def decode(tokens: list[int]) -> str:
    return "".join(chr(32 + t % PRINTABLE) for t in tokens)


def _messages_to_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"{m.get('role', 'user')}: {content}\n")
    return "".join(parts) + "assistant: "


class MockEngine:
    def __init__(self, target: ToyLM, drafter: Drafter | None = None, k: int = 5,
                 cost_model: CostModel | None = None, time_scale: float = 1.0,
                 model_name: str = "toy-target", confidence_threshold: float | None = None):
        self.target = target
        self.drafter_proto = drafter
        self.k = k
        self.cost = cost_model or CostModel()
        self.time_scale = time_scale
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.lock = threading.Lock()
        self.active = 0
        self.counters = {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0}
        self.per_pos = [0] * k

    def _sleep_for(self, ev: StepEvent, prompt_len: int) -> None:
        batch = max(1, self.active)
        if ev.prefill:
            ms = self.cost.forward_ms(1, prompt_len)
        elif self.drafter_proto is None:
            ms = self.cost.forward_ms(batch, 1)
        else:
            ms = (self.cost.spec_overhead_ms + self.cost.draft_step_ms(batch, self.drafter_proto.cost(self.k))
                  + self.cost.target_step_ms(batch, ev.drafted + 1))
        time.sleep(ms * self.time_scale / 1e3)

    def _steps(self, prompt: list[int], max_tokens: int, temperature: float, seed: int):
        rng = np.random.default_rng(seed)
        if self.drafter_proto is None:
            sess = self.target.session(prompt)
            for i in range(max_tokens):
                p = apply_temperature(sess.next_probs(), temperature)
                tok = int(np.argmax(p)) if temperature <= 0 else sample(p, rng)
                sess.extend([tok])
                yield StepEvent([tok], prefill=(i == 0))
            return
        import copy

        drafter = copy.copy(self.drafter_proto)  # per-request state (e.g. draft model cache)
        yield from speculative_stream(self.target, drafter, prompt, max_tokens, self.k, temperature, rng,
                                      self.confidence_threshold)

    def generate(self, text: str, max_tokens: int, temperature: float, seed: int = 0):
        """Yield decoded text pieces, one per engine step."""
        prompt = encode(text, self.target.vocab_size)
        with self.lock:
            self.active += 1
        try:
            for ev in self._steps(prompt, max_tokens, temperature, seed):
                self._sleep_for(ev, len(prompt))
                if not ev.prefill and self.drafter_proto is not None:
                    with self.lock:
                        self.counters["num_drafts"] += 1 if ev.drafted else 0
                        self.counters["num_draft_tokens"] += ev.drafted
                        self.counters["num_accepted_tokens"] += ev.accepted
                        for i in range(ev.accepted):
                            self.per_pos[i] += 1
                yield ev.tokens
        finally:
            with self.lock:
                self.active -= 1

    def metrics_text(self) -> str:
        if self.drafter_proto is None:
            return "# no speculative decoding configured\n"
        label = f'model_name="{self.model_name}"'
        lines = []
        for key, val in self.counters.items():
            lines += [f"# TYPE vllm:spec_decode_{key} counter", f"vllm:spec_decode_{key}_total{{{label}}} {val}"]
        lines.append("# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter")
        for i, val in enumerate(self.per_pos):
            lines.append(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{label},position="{i}"}} {val}')
        return "\n".join(lines) + "\n"


def make_handler(engine: MockEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep test output quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                body = {"object": "list", "data": [{"id": engine.model_name, "object": "model"}]}
                self._send(200, json.dumps(body).encode())
            elif self.path == "/metrics":
                self._send(200, engine.metrics_text().encode(), "text/plain; version=0.0.4")
            elif self.path == "/health":
                self._send(200, b"{}")
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            chat = self.path.rstrip("/") == "/v1/chat/completions"
            if not chat and self.path.rstrip("/") != "/v1/completions":
                self._send(404, b'{"error": "not found"}')
                return
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            if chat:
                text = _messages_to_text(req.get("messages", []))
            else:
                prompt = req.get("prompt", "")
                text = prompt if isinstance(prompt, str) else "".join(map(str, prompt))
            max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 64)
            temperature = float(req.get("temperature", 1.0))
            seed = int(req.get("seed", 0))
            prompt_tokens = len(encode(text, engine.target.vocab_size))
            obj = "chat.completion.chunk" if chat else "text_completion"

            def choice(piece: str, finish=None):
                if chat:
                    return {"index": 0, "delta": {"content": piece}, "finish_reason": finish}
                return {"index": 0, "text": piece, "finish_reason": finish}

            if not req.get("stream"):
                out = []
                for toks in engine.generate(text, max_tokens, temperature, seed):
                    out += toks
                content = decode(out)
                msg = ({"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}
                       if chat else {"index": 0, "text": content, "finish_reason": "length"})
                body = {"object": "chat.completion" if chat else "text_completion", "model": engine.model_name,
                        "choices": [msg], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(out),
                                                    "total_tokens": prompt_tokens + len(out)}}
                self._send(200, json.dumps(body).encode())
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            n_out = 0
            for toks in engine.generate(text, max_tokens, temperature, seed):
                n_out += len(toks)
                chunk = {"object": obj, "model": engine.model_name, "choices": [choice(decode(toks))]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            if (req.get("stream_options") or {}).get("include_usage"):
                usage = {"object": obj, "model": engine.model_name, "choices": [],
                         "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n_out,
                                   "total_tokens": prompt_tokens + n_out}}
                self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def serve(engine: MockEngine, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create (but do not start) the HTTP server; call ``serve_forever()`` on the result."""
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    server.daemon_threads = True
    return server
