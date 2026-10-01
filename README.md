# faster-llm-inference

**Speculative decoding, end to end: let a smaller model guess first.**

This repo turns the post *"5 resources for faster LLM inference"* into working code.
It has two halves:

1. **`specdec`, a reference implementation** that runs anywhere numpy does (no GPU).
   It includes every drafting method from the post: n-gram prompt lookup, draft models,
   EAGLE-style feature heads, and DFlash/DSpark-style block drafting with sequential
   correction. They all sit on top of a **lossless** verifier, and there is a
   **SPEED-Bench-style harness** that shows how acceptance and speedup move with workload,
   draft length and batch size.
2. **Production tooling for vLLM**: `--speculative-config` builders and serve scripts,
   a pipeline for training your own drafter with `vllm-project/speculators`, and a
   **traffic replayer** that measures speedup and acceptance on *your own* requests.

> Measure on your own traffic before trusting any speedup number. Everything in the
> toy half shows *mechanisms and trends*. The replay tool tells you what you actually get.

```
                ┌──────────── one target forward pass ────────────┐
 context ──► drafter guesses d1 d2 d3 d4 d5 ──► target scores all 6 positions at once
                                                   │
                       accept d1 d2 d3 ✓, reject d4 ✗ ──► emit d1 d2 d3 + corrected token
                                                   = 4 tokens for the price of ~1 pass
```

## The 5 resources → code

| # | Resource | What the post says | Where it lives here |
|---|----------|--------------------|---------------------|
| 1 | vLLM speculative decoding docs | Start with n-gram; drafts come from matches in your prompt; 5 speculative tokens, lookup of 4 | `specdec/drafters/ngram.py`, `configs/ngram.json`, `examples/01_ngram_prompt_lookup.py` |
| 2 | EAGLE 3.1 on the vLLM blog | Draft head on the target's hidden states; longer acceptance on long context; 2.03x at concurrency 1, 1.66x at 16 | `specdec/drafters/eagle.py`, `configs/eagle3.json`, `examples/02_eagle3_head.py` |
| 3 | SPEED-Bench (NVIDIA) | Test on diverse realistic prompts; synthetic inputs overestimate; best draft length changes with batch size | `specdec/bench/`, `specdec/cost_model.py`, `examples/03_speed_bench.py` |
| 4 | vllm-project/speculators | Train your own drafter (EAGLE-3, DFlash, DSpark, P-EAGLE, MTP); serve with plain `vllm serve` | `examples/04_train_drafter_speculators.sh`, `scripts/serve_speculator.sh` |
| 5 | DSpark for Kimi K3 | Block drafting + sequential correction; 4.11 tokens/round (6.42 on math) | `specdec/drafters/block.py`, `configs/dspark.json`, `examples/05_dspark_block_drafting.py` |

The numbers in the middle column are quoted from the post and its sources. This repo
does not reproduce them; see [docs/resources.md](docs/resources.md).

## Quickstart

```bash
pip install -e ".[dev]"          # only needs numpy (+ pytest for tests)
pytest -q                        # ~15 s, includes a statistical losslessness check

specdec demo                     # every drafter on one prompt: tau, acceptance, identical output
specdec bench --out results/     # SPEED-Bench-style sweep (~25 s)
specdec whatif --alpha 0.7       # roofline: speedup vs batch size and draft length
```

### Use it with vLLM

```bash
# 1) the post's starting point: n-gram, 5 speculative tokens, lookup of 4
specdec vllm-config ngram -k 5 --lookup-max 4 --target Qwen/Qwen3-8B
scripts/serve.sh Qwen/Qwen3-8B none 8000               # baseline
scripts/serve.sh Qwen/Qwen3-8B configs/ngram.json 8001  # speculative

# 2) replay YOUR traffic against both and compare (throughput, TPOT, TTFT, acceptance)
scripts/compare_on_traffic.sh my_traffic.jsonl \
    baseline=http://localhost:8000/v1 ngram=http://localhost:8001/v1

# 3) watch live acceptance from vLLM's Prometheus counters
specdec metrics --endpoint http://localhost:8001/v1 --watch 10
```

No GPU? `scripts/local_demo.sh` runs the same replay workflow against toy mock servers.

## What the toy benchmark shows

From [docs/results.md](docs/results.md): 8B dense on H100 under the roofline cost model,
toy target, k=5.

| drafter | realistic tau | synthetic tau | best k at B=1 | best k at B=256 |
|---|---|---|---|---|
| n-gram (k=5, lookup 4) | 2.66 | 4.14 | 8 | 4 |
| draft model | 4.38 | 4.96 | 8 | 2 |
| EAGLE-3 style | 4.75 | 5.22 | 8 | 2 |
| DFlash style | 3.87 | 4.88 | 8 | 2 |
| DSpark style | 4.00 | 4.95 | 8 | 2 |
| DSpark + adaptive verify | 3.91 | 4.89 | 8 | 4 |

* **Synthetic prompts overestimate acceptance** for every drafter, n-gram most of all (+56%).
* **The best draft length shrinks with batch size.** At high batch, verification becomes
  compute-bound and rejected drafts cost real FLOPs.
* **Acceptance depends heavily on workload.** On code edits n-gram is nearly free and
  nearly perfect. On open chat it barely helps.

Tau (τ) is the mean number of tokens emitted per target forward pass. Toy numbers are for
comparing methods and seeing trends, not for predicting production speedups.

## Repository layout

```
specdec/
  models.py          ToyLM: 2-layer RNN target + induction-style copy head; Session = KV cache
  verify.py          lossless speculative sampling (accept/reject + residual resampling)
  engine.py          draft -> verify -> accept loop, step streaming, vLLM-shaped stats
  ngram.py           suffix n-gram matching (prompt lookup)
  drafters/          ngram | draft_model | eagle (EAGLE-1/3 style) | block (DFlash/DSpark style)
  fit.py             tiny numpy trainers (ridge, softmax/logistic regression, low-rank SVD)
  cost_model.py      roofline model: why best k changes with batch size
  bench/             SPEED-Bench-style workloads + sweep + report
  serving/           vLLM configs, OpenAI-compatible replay client, /metrics parser, mock server
  cli.py             `specdec` command
configs/             ready-to-use --speculative-config JSON files
scripts/             serve / compare / local demo shell scripts
examples/            one runnable example per resource
data/                sample traffic JSONL (format reference)
docs/                architecture, algorithms, benchmarking, serving, training, results
tests/               pytest suite
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): system design, with diagrams
* [docs/resources.md](docs/resources.md): the 5 resources, what each claims, how this repo maps to them
* [docs/algorithms.md](docs/algorithms.md): the math behind speculative sampling and each drafter
* [docs/benchmarking.md](docs/benchmarking.md): how to measure speculation on your own traffic
* [docs/vllm-serving.md](docs/vllm-serving.md): choosing a method and k, and configuring vLLM
* [docs/training-drafters.md](docs/training-drafters.md): training drafters with speculators
* [docs/results.md](docs/results.md): toy benchmark results and how to read them
* [docs/cli.md](docs/cli.md): `specdec` command reference

## Caveats

* The toy target is a small RNN, not a transformer. The algorithms (verification, drafting
  logic, what each drafter learns from the target) are faithful. Absolute acceptance numbers
  are not comparable to real LLMs.
* The cost model is a roofline approximation: weights + KV reads vs FLOPs, plus fixed
  overheads. It ranks configurations and explains trends; it is not a simulator of vLLM.
* vLLM option names change between releases. The configs here follow the current vLLM docs.
  Pin your vLLM version and check `vllm serve --help` if a key is rejected.
