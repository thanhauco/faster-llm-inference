"""Replay traffic at several concurrency levels and compare endpoints.

Typical use: start one vLLM server without speculation and one with it (or
restart the same one), then::

    specdec replay --traffic my_traffic.jsonl \
        --endpoint baseline=http://localhost:8000/v1 \
        --endpoint ngram=http://localhost:8001/v1 \
        --concurrency 1,4,16,64

For every endpoint and concurrency this reports output tokens/s, per-user
tokens/s, TTFT/TPOT percentiles and, when the server exposes spec-decode
metrics, the acceptance length measured on exactly this traffic. Speedups are
computed against the first endpoint.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass

from specdec.serving import prom
from specdec.serving.client import Request, RequestResult, list_models, send


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


@dataclass
class LoadResult:
    endpoint: str
    concurrency: int
    results: list[RequestResult]
    wall_s: float
    spec: prom.SpecDecodeSnapshot | None = None

    def summary(self) -> dict:
        ok = [r for r in self.results if r.ok]
        out_tokens = sum(r.output_tokens for r in ok)
        tpots = [r.tpot_s for r in ok if r.tpot_s is not None]
        per_user = [(r.output_tokens - 1) / (r.latency_s - r.ttft_s) for r in ok
                    if r.output_tokens > 1 and r.latency_s > r.ttft_s]
        chunk_ratio = [r.tokens_per_chunk for r in ok if r.tokens_per_chunk]
        s = {
            "endpoint": self.endpoint,
            "concurrency": self.concurrency,
            "requests": len(self.results),
            "errors": len(self.results) - len(ok),
            "output_tokens": out_tokens,
            "wall_s": round(self.wall_s, 3),
            "output_tok_s": out_tokens / self.wall_s if self.wall_s > 0 else 0.0,
            "per_user_tok_s_mean": sum(per_user) / len(per_user) if per_user else None,
            "ttft_ms_p50": _ms(percentile([r.ttft_s for r in ok], 50)),
            "ttft_ms_p90": _ms(percentile([r.ttft_s for r in ok], 90)),
            "ttft_ms_p99": _ms(percentile([r.ttft_s for r in ok], 99)),
            "tpot_ms_p50": _ms(percentile(tpots, 50)),
            "tpot_ms_p90": _ms(percentile(tpots, 90)),
            "tpot_ms_p99": _ms(percentile(tpots, 99)),
            "latency_s_p50": percentile([r.latency_s for r in ok], 50),
            "tokens_per_chunk_mean": sum(chunk_ratio) / len(chunk_ratio) if chunk_ratio else None,
        }
        if self.spec is not None and self.spec.found:
            s["spec_decode"] = self.spec.to_dict()
        if len(ok) < len(self.results):
            s["first_error"] = next(r.error for r in self.results if not r.ok)
        return s


def _ms(x: float | None) -> float | None:
    return None if x is None else x * 1e3


def run_load(base_url: str, model: str, requests: list[Request], concurrency: int, api_key: str | None = None,
             timeout: float = 600.0, scrape_metrics: bool = True, label: str | None = None) -> LoadResult:
    """Closed-loop load: ``concurrency`` workers each send the next request as soon as theirs finishes."""
    before = _try_scrape(base_url) if scrape_metrics else None
    work: queue.Queue = queue.Queue()
    for i, r in enumerate(requests):
        work.put((i, r))
    results: list[RequestResult | None] = [None] * len(requests)

    def worker():
        while True:
            try:
                i, r = work.get_nowait()
            except queue.Empty:
                return
            results[i] = send(base_url, model, r, api_key=api_key, timeout=timeout)

    start = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    after = _try_scrape(base_url) if scrape_metrics else None
    spec = after - before if (before is not None and after is not None) else None
    return LoadResult(label or base_url, concurrency, [r for r in results if r is not None], wall, spec)


def _try_scrape(base_url: str) -> prom.SpecDecodeSnapshot | None:
    try:
        return prom.scrape(base_url)
    except Exception:
        return None


def sweep(endpoints: dict[str, str], requests: list[Request], concurrencies: list[int], model: str | None = None,
          api_key: str | None = None, warmup: int = 2, log=print) -> dict:
    """Run every endpoint at every concurrency; speedups are relative to the first endpoint."""
    summaries = []
    for label, url in endpoints.items():
        served = model or list_models(url, api_key)[0]
        if warmup:
            run_load(url, served, requests[:warmup], 1, api_key, scrape_metrics=False)
        for c in concurrencies:
            res = run_load(url, served, requests, c, api_key, label=label)
            s = res.summary()
            summaries.append(s)
            acc = s.get("spec_decode", {}).get("mean_acceptance_length")
            log(f"  {label:<14} c={c:<4} {s['output_tok_s']:9.1f} tok/s  "
                f"tpot p50={_fmt(s['tpot_ms_p50'])}ms  ttft p50={_fmt(s['ttft_ms_p50'])}ms"
                + (f"  tau={acc:.2f}" if acc else "") + (f"  errors={s['errors']}" if s["errors"] else ""))
    base_label = next(iter(endpoints))
    base = {s["concurrency"]: s for s in summaries if s["endpoint"] == base_label}
    for s in summaries:
        b = base.get(s["concurrency"])
        if b and b["output_tok_s"]:
            s["throughput_speedup"] = s["output_tok_s"] / b["output_tok_s"]
            if s["tpot_ms_p50"] and b["tpot_ms_p50"]:
                s["tpot_speedup"] = b["tpot_ms_p50"] / s["tpot_ms_p50"]
    return {"baseline": base_label, "concurrencies": concurrencies, "results": summaries}


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def to_markdown(report: dict) -> str:
    lines = [
        "# Replay results on your traffic",
        "",
        f"Baseline endpoint: `{report['baseline']}`. Speedups are relative to it at the same concurrency.",
        "",
        "| endpoint | concurrency | output tok/s | speedup | per-user tok/s | TPOT p50 ms | TTFT p50 ms | tau |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in report["results"]:
        tau = s.get("spec_decode", {}).get("mean_acceptance_length")
        lines.append(
            f"| {s['endpoint']} | {s['concurrency']} | {s['output_tok_s']:.1f} | "
            f"{s.get('throughput_speedup', 1.0):.2f}x | {_fmt(s['per_user_tok_s_mean'])} | "
            f"{_fmt(s['tpot_ms_p50'])} | {_fmt(s['ttft_ms_p50'])} | {'-' if tau is None else f'{tau:.2f}'} |"
        )
    return "\n".join(lines) + "\n"


def save(report: dict, path_prefix: str) -> None:
    with open(path_prefix + ".json", "w") as f:
        json.dump(report, f, indent=1)
    with open(path_prefix + ".md", "w") as f:
        f.write(to_markdown(report))
