# Measuring speculative decoding on your own traffic

> Measure on your own traffic before trusting any speedup number.

Published speedups depend on the model, the drafter, the hardware, the batch size,
sampling settings and, above all, the text being generated. This guide shows how to get
the number that matters: *your* speedup on *your* requests.

## 1. Collect a traffic sample

Export a few hundred real requests (more is better; at least a few per category you serve)
to JSONL, one object per line. Any of these shapes works (see `data/sample_traffic.jsonl`):

```json
{"prompt": "def fibonacci(n):", "max_tokens": 128, "temperature": 0.0}
{"messages": [{"role": "user", "content": "..."}], "max_tokens": 512, "temperature": 0.7}
{"body": {"model": "...", "messages": [...], "max_tokens": 256, "top_p": 0.9}}
```

* Keep the real `max_tokens` and `temperature`. Acceptance drops with temperature.
* Keep the category mix. Code edits and RAG speculate well; open chat does not.
* Strip PII as your policy requires; the content still has to be realistic.

## 2. Start a baseline and a speculative server

Same GPU type, same vLLM version, same flags apart from `--speculative-config`:

```bash
scripts/serve.sh Qwen/Qwen3-8B none 8000
scripts/serve.sh Qwen/Qwen3-8B configs/ngram.json 8001
scripts/serve_speculator.sh RedHatAI/Qwen3-8B-speculator.eagle3 8002
```

If you only have one GPU, run them one after another on the same port. `replay` accepts any
number of `label=url` endpoints.

## 3. Replay at the concurrencies you actually run

```bash
specdec replay --traffic my_traffic.jsonl \
  --endpoint baseline=http://localhost:8000/v1 \
  --endpoint ngram=http://localhost:8001/v1 \
  --endpoint eagle3=http://localhost:8002/v1 \
  --concurrency 1,4,16,64 --out results/replay
```

Per endpoint and concurrency the report has:

| Column | Meaning |
|---|---|
| output tok/s | total generated tokens / wall time (system throughput) |
| speedup | throughput relative to the first endpoint at the same concurrency |
| per-user tok/s | mean of `(tokens − 1) / (latency − TTFT)` per request (what a user feels) |
| TPOT p50 | median time per output token after the first |
| TTFT p50 | median time to first token (speculation should not change this much) |
| tau | mean acceptance length from the server's `/metrics`, diffed around the run |

Concurrency is **closed-loop**: N workers each keep one request in flight, matching vLLM's
`--max-concurrency` benchmark mode.

## 4. Read acceptance from the server

```bash
specdec metrics --endpoint http://localhost:8001/v1             # since server start
specdec metrics --endpoint http://localhost:8001/v1 --watch 30  # rolling 30 s windows
```

This parses vLLM's Prometheus counters:

| Counter | Meaning |
|---|---|
| `vllm:spec_decode_num_drafts` | verification steps that had drafts |
| `vllm:spec_decode_num_draft_tokens` | proposed draft tokens |
| `vllm:spec_decode_num_accepted_tokens` | accepted draft tokens |
| `vllm:spec_decode_num_accepted_tokens_per_pos{position=i}` | accepted at draft position i |

`τ = 1 + accepted / drafts`. Per-position acceptance shows where your drafts stop paying
off: if position 4 is accepted 10% of the time, k=3 will probably be faster than k=5 at
high load. Newer vLLM versions can also return acceptance per request
(`--per-request-spec-decode-metrics summary`), which is useful for breaking τ down by category.

## 5. Pitfalls

| Pitfall | Why it misleads | What to do |
|---|---|---|
| Synthetic prompts (random ids, repeated filler) | Degenerate inputs produce degenerate, repetitive outputs that are easy to draft. The toy bench shows synthetic τ above realistic τ for every drafter. | Replay real traffic. |
| Measuring only at concurrency 1 | Speedup shrinks as the batch becomes compute-bound. | Sweep up to your peak concurrency. |
| Fixed k across load levels | Best k at batch 1 is often too long at batch 64+. | Re-tune k per deployment, or use dynamic speculation / adaptive verification. |
| Inter-chunk latency as ITL | One SSE chunk can carry several accepted tokens. | Use TPOT from token counts (`replay` does). |
| `ignore_eos` / forced long outputs | Changes the text distribution, usually toward repetition. | Use real `max_tokens` and let EOS happen. |
| Greedy-only testing | Acceptance falls with temperature. | Use production sampling params. |
| Cold servers | First requests include compilation and CUDA graph capture. | `replay` sends warm-up requests (`--warmup`). |
| Prefix caching differences | Changes TTFT and load; can mask or exaggerate gains. | Same flags on both servers. |
| Quoting τ without k | τ = 3 at k=3 and at k=8 are very different configurations. | Always report k, τ and per-position acceptance. |

## 6. Before you deploy: the lab

The toy lab is useful *before* you have GPUs, or to build intuition:

```bash
specdec bench --out results/                   # which method, which k, synthetic vs realistic
specdec whatif --alpha 0.65 --model-size 70b   # what α do I need for a speedup at B=64?
```

`whatif` takes a per-token acceptance α. You can estimate α from a real server's
per-position acceptance (roughly the ratio between consecutive positions) and then explore
k and batch size on the roofline model for your GPU.

## 7. GPU-free dry run

```bash
scripts/local_demo.sh
```

This starts three toy OpenAI-compatible servers (baseline, n-gram, EAGLE-3 style), replays
`data/sample_traffic.jsonl` and prints the same report you would get from vLLM.
