# The 5 resources

The post that started this repo:

> **5 resources for faster LLM inference.** The trick is to let a smaller model guess first.
> ... Measure on your own traffic before trusting any speedup number.

Below, each resource appears with what the post claims, the upstream sources, what this
repo implements, and how to run it. **Numbers attributed to the post are quoted, not
reproduced.** The toy lab demonstrates the mechanisms and the direction of each effect.
Your production numbers come from `specdec replay` on your own traffic.

Original short links from the post: [1](https://lnkd.in/d23kzBKQ) ·
[2](https://lnkd.in/drrBVz4J) · [3](https://lnkd.in/dENXCmvZ) ·
[4](https://lnkd.in/dKrCdJaZ) · [5](https://lnkd.in/d7hy9q-9)

---

## 1. vLLM speculative decoding docs: start with n-gram

**Post:** start with n-gram, the simplest method. Drafts come from matches in your prompt.
Config: 5 speculative tokens, lookup of 4.

**Upstream:** [vLLM docs: Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
(the N-Gram page uses exactly `{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}`).
vLLM also documents suffix decoding, EAGLE, MTP, draft models, PARD, MLP speculators,
dynamic speculative decoding and adaptive verification.

**Here:**

| | |
|---|---|
| Algorithm | `specdec/ngram.py` (suffix match), `specdec/drafters/ngram.py` (`NgramDrafter.vllm_default()`) |
| vLLM config | `configs/ngram.json`, `specdec vllm-config ngram -k 5 --lookup-max 4` |
| Example | `python examples/01_ngram_prompt_lookup.py` |

**What you will see:** n-gram is close to free (no draft model, CPU-side matching) and
excellent when outputs repeat inputs: on the toy code-edit workload τ is about 4.5 to 5.8 at
k=5. It barely helps open-ended chat (τ ≈ 1.5), and when nothing matches it proposes
nothing, so the step costs about the same as normal decoding.

---

## 2. EAGLE 3.1 on the vLLM blog: draft heads on the target's hidden states

**Post:** draft head built on the target's hidden states; up to 2x longer acceptance on long
context; 2.03x throughput at concurrency 1, 1.66x at 16.

**Upstream:** [vLLM blog](https://blog.vllm.ai/); [EAGLE paper](https://arxiv.org/abs/2401.15077);
[SafeAI Lab EAGLE repo](https://github.com/SafeAILab/EAGLE); vLLM EAGLE docs
(`"method": "eagle3"`). Pre-trained heads:
[RedHatAI speculator models](https://huggingface.co/collections/RedHatAI/speculator-models).

**Here:**

| | |
|---|---|
| Algorithm | `specdec/drafters/eagle.py`: `layers="top"` (EAGLE-1 style) vs `layers="all"` (EAGLE-3 style multi-layer fusion), `ttt_rounds` (training-time test) |
| vLLM config | `configs/eagle3.json`, preset `eagle3-llama3.1-8b` / `eagle3-qwen3-8b` |
| Example | `python examples/02_eagle3_head.py` |

**What you will see:** fusing all layers and training on the head's own rollouts lifts
acceptance at *every* draft position. On the toy target, per-position acceptance at
position 6 goes from ~0.49 (top layer) to ~0.65 (fused + TTT). The drop in speedup from
concurrency 1 to 16 quoted in the post is the batch-size effect explained in resource 3
and [ARCHITECTURE.md §10](ARCHITECTURE.md#10-cost-model).

---

## 3. SPEED-Bench (NVIDIA): evaluate drafters on realistic, diverse prompts

**Post:** tests drafters on diverse, realistic prompts; synthetic inputs overestimate
throughput; best draft length changes with batch size.

**Here:**

| | |
|---|---|
| Workloads | `specdec/bench/workloads.py`: code-edit, rag-qa, multi-turn-chat, open-chat, reasoning (realistic) vs synthetic-random, synthetic-repeat |
| Sweep | `specdec/bench/speed_bench.py`: drafters × workloads × k × batch |
| Cost model | `specdec/cost_model.py` (roofline: memory- vs compute-bound) |
| Example | `python examples/03_speed_bench.py`, `specdec bench`, `specdec whatif` |

**What you will see** ([results.md](results.md)): synthetic prompts inflate τ for all
seven drafter configurations, n-gram most (+56%). The best k falls from 8 at batch 1 to
2-4 at batch 256. For the same drafter, acceptance swings widely by category: n-gram τ is 4.45 on code edits and 1.46 on open chat.

---

## 4. vllm-project/speculators: train your own drafter

**Post:** train your own drafter for vLLM (EAGLE-3, DFlash, DSpark, P-EAGLE, MTP); serve it
with a plain `vllm serve`.

**Upstream:** [github.com/vllm-project/speculators](https://github.com/vllm-project/speculators),
[docs](https://docs.vllm.ai/projects/speculators/en/latest/). Pipeline: `speculators
prepare-data` → hidden-state extraction from vLLM (online, offline or hybrid) →
`torchrun -m speculators.train --speculator-type ...` → `vllm serve <checkpoint>`.
The checkpoint is self-describing: vLLM reads `speculators_config` from `config.json`.

**Here:**

| | |
|---|---|
| End-to-end script | `examples/04_train_drafter_speculators.sh` (`ALGO=eagle3|peagle|dflash|dspark`) |
| Serve a checkpoint | `scripts/serve_speculator.sh RedHatAI/Qwen3-8B-speculator.eagle3` |
| Concepts in miniature | `specdec/drafters/traces.py` (on-policy traces), `EagleDrafter.fit`, `BlockDrafter.fit` |
| Guide | [training-drafters.md](training-drafters.md) |

---

## 5. DSpark for Kimi K3: block drafting plus sequential correction

**Post:** block drafting plus sequential correction; 4.11 tokens accepted per round, 6.42
on math; ~110 to ~435 tok/s per user on math.

**Upstream:** DSpark is described in the speculators docs as DFlash's block-parallel
backbone plus (a) a low-rank **Markov head** that biases each draft position by the token
before it, and (b) a **confidence head** predicting per-position acceptance, used for
adaptive verification. Paper: *DSpark: Confidence-Scheduled Speculative Decoding with
Semi-Autoregressive Generation* ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)).
vLLM serves it with `"method": "dspark"`.

**Here:**

| | |
|---|---|
| Algorithm | `specdec/drafters/block.py`: `BlockDrafter.dflash(...)`, `BlockDrafter.dspark(...)` |
| Adaptive verification | `speculative_generate(..., confidence_threshold=0.3)` |
| vLLM config | `configs/dspark.json` |
| Example | `python examples/05_dspark_block_drafting.py` |

**What you will see:** one drafting pass per block makes block drafters the cheapest
model-based option at small batch. On open-ended text, pure block drafting decays fast
along the block, and the Markov correction recovers part of it (toy τ 2.56 → 3.11). The
confidence head lets the engine stop verifying where the block stops being worth it, which
keeps speedup at large batch (toy B=256 open-chat: 0.96x → 1.94x). Like the post's math
numbers, acceptance is much higher on low-entropy "reasoning" text than on open chat.

---

## The sixth resource: your own traffic

> Measure on your own traffic before trusting any speedup number.

`specdec replay` + `specdec metrics` ([benchmarking.md](benchmarking.md)): replay a JSONL
sample of real requests against baseline and speculative servers at your real
concurrency, then read τ from the server's own counters.
