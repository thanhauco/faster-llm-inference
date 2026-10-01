# `specdec` CLI reference

Install with `pip install -e .`, then run `specdec <command> --help` for every flag.
`python -m specdec ...` works too.

| Command | Purpose |
|---|---|
| `demo` | Run every drafter on one prompt; print τ, acceptance and whether output equals the baseline |
| `bench` | SPEED-Bench-style sweep on the toy target; writes `.md`, `.json` and `_rows.csv` |
| `whatif` | Roofline speedup table for a per-token acceptance rate across k and batch size |
| `vllm-config` | Print `--speculative-config` JSON, the `vllm serve` command and an offline snippet |
| `replay` | Replay a JSONL traffic sample against one or more OpenAI-compatible endpoints |
| `metrics` | Read spec-decode acceptance from a vLLM `/metrics` endpoint (optionally watch) |
| `mock-server` | Toy OpenAI-compatible server (baseline or speculative) for GPU-free testing |

## demo

```bash
specdec demo                         # default workload: reasoning (greedy)
specdec demo --workload code-edit -k 8
```

## bench

```bash
specdec bench --out results/
specdec bench --drafters ngram,eagle3,dspark --workloads code-edit,open-chat \
  --ks 1,2,3,4,6,8 --batch-sizes 1,8,32,128 --prompts 8 \
  --hardware a100 --model-size 70b --context-len 4096 --out results/a100
```

Drafters: `ngram, draft-model, eagle1, eagle3, dflash, dspark, dspark-adaptive`.
Workloads: `code-edit, rag-qa, multi-turn-chat, open-chat, reasoning, synthetic-random, synthetic-repeat`.
Hardware: `h100, a100, l40s, b200`. Model sizes: `8b, 70b, moe-30b-a3b`.

## whatif

```bash
specdec whatif --alpha 0.7                                      # EAGLE-like (k draft passes)
specdec whatif --alpha 0.8 --draft-passes 1 --draft-rel-params 0.1   # block drafter
specdec whatif --alpha 0.5 --draft-passes 0 --draft-rel-params 0     # n-gram
```

## vllm-config

```bash
specdec vllm-config ngram -k 5 --lookup-max 4 --target Qwen/Qwen3-8B
specdec vllm-config eagle3 -k 3 --model RedHatAI/Qwen3-8B-speculator.eagle3 --target Qwen/Qwen3-8B
specdec vllm-config draft_model -k 5 --model Qwen/Qwen3-0.6B --target Qwen/Qwen3-8B
specdec vllm-config mtp -k 1 --target XiaomiMiMo/MiMo-7B-Base
specdec vllm-config speculators --model ./output/checkpoints/checkpoint_best
specdec vllm-config preset --preset eagle3-llama3.1-8b
```

Presets: `ngram-post, draft-qwen3-0.6b, eagle3-llama3.1-8b, eagle3-qwen3-8b, dflash-qwen3-8b, mtp-mimo-7b`.

## replay

```bash
specdec replay --traffic my_traffic.jsonl \
  --endpoint baseline=http://localhost:8000/v1 --endpoint spec=http://localhost:8001/v1 \
  --concurrency 1,4,16,64 --limit 500 --shuffle --out results/replay
```

| Flag | Meaning |
|---|---|
| `--endpoint label=url` | repeatable; the first one is the baseline for speedups |
| `--concurrency` | closed-loop in-flight requests per run |
| `--model` | served model name (default: first entry of `/v1/models`, per endpoint) |
| `--max-tokens`, `--temperature` | override the values in the traffic file |
| `--warmup N` | warm-up requests per endpoint before measuring |
| `--api-key` | sent as a bearer token |

## metrics

```bash
specdec metrics --endpoint http://localhost:8001/v1
specdec metrics --endpoint http://localhost:8001/v1 --watch 10
```

## mock-server

```bash
specdec mock-server --port 8000                                   # baseline
specdec mock-server --port 8001 --method ngram -k 5
specdec mock-server --port 8002 --method dspark-adaptive -k 8 --time-scale 0.5
```

`--time-scale` multiplies the simulated per-step latency (0 for no sleeping).
