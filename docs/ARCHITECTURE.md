# Architecture

This document explains how `faster-llm-inference` is put together: the components, how
data flows between them, and the design decisions behind them. Diagrams use
[Mermaid](https://mermaid.js.org/), which GitHub renders inline.

- [1. Goals](#1-goals)
- [2. System overview](#2-system-overview)
- [3. Module map](#3-module-map)
- [4. The speculative decoding step](#4-the-speculative-decoding-step)
- [5. Lossless verification](#5-lossless-verification)
- [6. Sessions, the pending token and rollback](#6-sessions-the-pending-token-and-rollback)
- [7. Drafters](#7-drafters)
- [8. Training drafters on target traces](#8-training-drafters-on-target-traces)
- [9. Benchmark pipeline (SPEED-Bench style)](#9-benchmark-pipeline-speed-bench-style)
- [10. Cost model](#10-cost-model)
- [11. Production workflow with vLLM](#11-production-workflow-with-vllm)
- [12. Mock server](#12-mock-server)
- [13. Extension points](#13-extension-points)
- [14. Design decisions](#14-design-decisions)

---

## 1. Goals

| Goal | How |
|---|---|
| Show *how* each drafting method works, in code you can step through | Pure-numpy implementations of n-gram, draft-model, EAGLE-style, DFlash/DSpark-style drafters |
| Never trade correctness for speed | One shared verifier implementing exact speculative sampling; tested statistically |
| Explain *when* speculation pays off | SPEED-Bench-style workloads + roofline cost model across draft length and batch size |
| Make real deployments measurable | vLLM config builders, traffic replay client, Prometheus spec-decode metrics |
| Run anywhere | Only dependency is numpy; HTTP via stdlib; mock server for GPU-free end-to-end runs |

Non-goals: being a fast inference engine, or reproducing published speedup numbers.
The toy target model is for mechanisms, not absolute numbers.

---

## 2. System overview

The repo has two halves that share vocabulary and metrics: a **lab** (left), where every
algorithm runs on a toy target, and a **production toolkit** (right), which drives real
vLLM servers.

```mermaid
flowchart LR
    subgraph LAB["Lab: specdec (numpy, no GPU)"]
        direction TB
        TM["ToyLM target<br/>2-layer RNN + copy head"]
        TR["Target traces<br/>tokens, hidden states, probs"]
        DR["Drafters<br/>ngram | draft model | EAGLE | DFlash/DSpark"]
        EN["Engine<br/>draft → verify → accept"]
        VF["Verifier<br/>lossless speculative sampling"]
        BN["SPEED-Bench-style harness<br/>workloads × drafters × k"]
        CM["Roofline cost model<br/>batch × tokens → ms"]
        RP1["Report<br/>tau, speedup, best k per batch"]
        TM --> TR --> DR
        DR --> EN
        TM --> EN
        EN <--> VF
        BN --> EN
        EN -->|SpecStats| BN
        BN --> CM --> RP1
    end

    subgraph PROD["Production toolkit: specdec.serving"]
        direction TB
        CFG["Config builders<br/>--speculative-config JSON"]
        VS["vLLM servers<br/>baseline + speculative"]
        TF["Your traffic<br/>JSONL"]
        RC["Replay client<br/>closed-loop concurrency"]
        PM["/metrics parser<br/>vllm:spec_decode_*"]
        RP2["Replay report<br/>tok/s, TPOT, TTFT, tau"]
        SP["speculators<br/>train EAGLE-3 / DFlash / DSpark"]
        CFG --> VS
        SP -->|checkpoint| VS
        TF --> RC --> VS
        VS --> PM --> RP2
        RC --> RP2
    end

    MOCK["Mock server<br/>ToyLM + drafter behind<br/>an OpenAI-compatible API"]
    EN -.-> MOCK -.->|stands in for| VS
```

---

## 3. Module map

```mermaid
flowchart TD
    cli["cli.py"] --> bench["bench/speed_bench.py"]
    cli --> serving
    bench --> workloads["bench/workloads.py"]
    bench --> engine["engine.py"]
    bench --> cost["cost_model.py"]
    bench --> drafters
    engine --> verify["verify.py"]
    engine --> models["models.py"]
    engine --> sampling["sampling.py"]
    subgraph drafters["drafters/"]
        base["base.py<br/>Drafter, Draft, DraftCost"]
        ng["ngram.py"]
        dm["draft_model.py"]
        eg["eagle.py"]
        bl["block.py"]
        tr["traces.py"]
    end
    ng --> ngram["ngram.py (suffix match)"]
    models --> ngram
    eg --> fit["fit.py"]
    bl --> fit
    eg --> tr
    bl --> tr
    subgraph serving["serving/"]
        cfg["configs.py"]
        cl["client.py"]
        rp["replay.py"]
        pr["prom.py"]
        tf["traffic.py"]
        ms["mock_server.py"]
        ht["http.py"]
    end
    rp --> cl --> ht
    rp --> pr --> ht
    ms --> engine
    ms --> cost
```

| Module | Responsibility |
|---|---|
| `models.py` | `ToyLM` target (and draft-model factory), `Session` incremental state with rollback |
| `ngram.py` | Longest-suffix n-gram match, shared by the n-gram drafter and the toy target's copy head |
| `verify.py` | `verify_draft`: accept/reject drafts so output ~ target distribution exactly |
| `engine.py` | `speculative_stream` / `speculative_generate` loop; `SpecStats` with vLLM-shaped metrics |
| `drafters/` | One class per method behind the `Drafter` interface; `traces.py` builds training data |
| `fit.py` | Small numpy trainers: ridge, softmax and logistic regression (Adam), truncated SVD |
| `cost_model.py` | Roofline estimate of forward-pass time; converts acceptance into tokens/s and speedup |
| `bench/` | Workload generators (realistic vs synthetic) and the sweep/report pipeline |
| `serving/` | vLLM config builders, streaming client, replay sweeps, `/metrics` parsing, mock server |
| `cli.py` | `specdec` entry point (`demo`, `bench`, `whatif`, `vllm-config`, `replay`, `metrics`, `mock-server`) |

---

## 4. The speculative decoding step

Every step costs one drafting phase plus **one** target forward pass over `[pending, d1..dn]`,
and always emits at least one token.

```mermaid
sequenceDiagram
    autonumber
    participant E as Engine
    participant D as Drafter
    participant T as Target (Session)
    participant V as Verifier

    Note over E,T: Prefill: target scores the prompt and samples the first token (the "pending" token)
    loop until max_new_tokens
        E->>D: propose(session, k, temperature)
        D-->>E: Draft(tokens d1..dn, probs q1..qn, confidence?)
        opt drafter reports confidence (DSpark) and a threshold is set
            E->>E: truncate draft where cumprod(confidence) < threshold
        end
        E->>T: extend([d1..dn]) and read p1..p(n+1)
        Note right of T: one forward pass over n+1 positions
        E->>V: verify_draft(d, q, p)
        V-->>E: num_accepted = a, next_token
        E->>T: truncate(base + a), then extend([next_token])
        Note right of T: rejected drafts rolled back (KV cache)
        E-->>E: emit d1..da + next_token, record (n, a) in SpecStats
    end
```

`SpecStats` records exactly what vLLM reports per request: number of steps, drafted and
accepted tokens, the acceptance histogram and per-position acceptance. It also keeps a
per-step log, so one run at `k_max` can be re-scored for any smaller `k`
(`SpecStats.truncated`).

---

## 5. Lossless verification

```mermaid
flowchart TD
    S([for i = 1..n]) --> G{greedy?}
    G -->|yes| GM{"d_i == argmax p_i ?"}
    GM -->|yes| NX[accept, next i]
    GM -->|no| GR["emit argmax p_i, stop"]
    G -->|no| R{"u < min(1, p_i(d_i) / q_i(d_i)) ?"}
    R -->|yes| NX
    R -->|no| RS["emit sample from norm(max(p_i − q_i, 0)), stop"]
    NX --> S
    S -->|all n accepted| B["emit bonus token from p_(n+1)"]
```

* Deterministic drafters (n-gram) pass `probs=None`. The verifier treats `q_i` as one-hot on
  `d_i`, so acceptance is `p_i(d_i)` and the residual is `p_i` with `d_i` zeroed out.
* Temperature is applied in probability space (`p^(1/T)`, renormalised) for both target and
  drafter before verification. This is valid for mixture models like `ToyLM`.
* `tests/test_verify.py` enumerates every 3-token continuation of a small target and checks
  that the empirical distribution under speculation matches the exact target distribution.

---

## 6. Sessions, the pending token and rollback

`Session` is the toy KV cache: it stores per-token hidden states and supports
`extend`, `truncate` and `sync` (reuse the longest common prefix).

```mermaid
stateDiagram-v2
    [*] --> Prefilled: session(prompt) + sample first token
    Prefilled --> Drafting: pending token = last emitted token
    Drafting --> Scored: extend(d1..dn)<br/>one target pass → n+1 distributions
    Scored --> RolledBack: truncate(base + accepted)
    RolledBack --> Prefilled: extend(next_token)<br/>(new pending token)
    Prefilled --> [*]: max_new_tokens reached
```

**The pending-token rule.** After each step, the newly emitted token has been *sampled*,
but a real target has not yet run a forward pass on it. That happens at the start of the
next verification pass. The toy session has computed its state already, so drafters that
reuse target hidden states (EAGLE, DFlash/DSpark) must read features only up to
`session.states[-2]` and take the pending token as an input embedding. Otherwise the first
draft token would be "free", and acceptance would be inflated in a way that cannot happen
in vLLM.

```
tokens:   ... x_(t-1)   x_t (pending)  | d1  d2  d3 ...
features: ... f_(t-1)  [not yet real]  |
               ▲           ▲
               │           └── drafter input: embedding(x_t)
               └────────────── drafter input: target features (EAGLE / DFlash anchor)
```

---

## 7. Drafters

```mermaid
classDiagram
    class Drafter {
        <<abstract>>
        +name: str
        +reset()
        +propose(session, k, temperature, rng) Draft
        +cost(k) DraftCost
    }
    class Draft {
        +tokens: list~int~
        +probs: ndarray | None
        +confidence: ndarray | None
        +truncate(n) Draft
    }
    class DraftCost {
        +passes: int
        +tokens_per_pass: int
        +rel_params: float
        +cpu_ms: float
    }
    class NgramDrafter {
        +prompt_lookup_min
        +prompt_lookup_max
        +vllm_default()
    }
    class DraftModelDrafter {
        +model: ToyLM
        -_session: Session
    }
    class EagleDrafter {
        +layers: "all" | "top"
        +A: feature regressor
        +fit(traces, ttt_rounds)
    }
    class BlockDrafter {
        +block_size
        +heads: per-position backbone
        +markov_A, markov_B: low-rank bias
        +conf_w: confidence head
        +dflash()
        +dspark()
        +fit(traces)
    }
    Drafter <|-- NgramDrafter
    Drafter <|-- DraftModelDrafter
    Drafter <|-- EagleDrafter
    Drafter <|-- BlockDrafter
    Drafter ..> Draft : returns
    Drafter ..> DraftCost : declares
```

| Drafter | Mirrors vLLM `method` | Proposal | Draft passes per step | Learns from target |
|---|---|---|---|---|
| `NgramDrafter` | `ngram` | tokens after the longest earlier match of the suffix | 0 (CPU only) | nothing |
| `DraftModelDrafter` | `draft_model` | small LM sampled autoregressively | k | (pre-trained, same family) |
| `EagleDrafter(layers="top")` | `eagle` | extrapolate top-layer features, decode with target LM head | k | hidden states |
| `EagleDrafter(layers="all")` | `eagle3` | same, with fused multi-layer features + training-time test | k | hidden states (all layers) |
| `BlockDrafter.dflash` | `dflash` | whole block from one anchored pass | 1 | hidden states + tokens |
| `BlockDrafter.dspark` | `dspark` | block + Markov correction + confidence | 1 | hidden states + tokens + acceptance |

### 7.1 N-gram (prompt lookup)

```mermaid
flowchart LR
    C["context tokens<br/>(prompt + generated)"] --> M["find longest suffix n-gram<br/>min ≤ n ≤ max (lookup of 4)<br/>that occurred earlier"]
    M -->|match at position j| P["propose tokens j+1 .. j+k"]
    M -->|no match| E["empty draft → plain decode step"]
```

### 7.2 EAGLE-style feature head

```mermaid
flowchart LR
    F["target features f_(t-1)<br/>EAGLE-3: concat of all layers"] --> H
    X["embedding(pending x_t)"] --> H["head: f̂_t = tanh(A·[f; e; 1])"]
    H --> LM["target LM head (frozen, shared)"] --> Q["q_1 → sample d1"]
    H -->|f̂_t| H2["head step 2: f̂_(t+1) = head(f̂_t, e(d1))"] --> LM2["LM head"] --> Q2["q_2 → d2"]
    H2 -.->|... k steps| HK["..."]
```

Training (`EagleDrafter.fit`): ridge regression from `[f_(t-1), e(x_t)]` to `atanh(f_t)`
on target traces. With `ttt_rounds > 0` it adds samples whose inputs are the head's
*own* multi-step predictions ("training-time test"), so it learns to correct its own drift.

### 7.3 DFlash / DSpark block drafting

```mermaid
flowchart LR
    F["target features f_(t-1)"] --> BB
    A["embedding(anchor = pending token)"] --> BB["block backbone: ONE pass<br/>logits for positions 1..B"]
    BB --> L1["pos 1 logits"] & L2["pos 2 logits"] & LB["pos B logits"]
    subgraph SEQ["DSpark sequential correction (cheap)"]
        direction LR
        M1["+ Markov bias A[x_t]·B"] --> S1["sample d1"] --> M2["+ Markov bias A[d1]·B"] --> S2["sample d2"] --> MB["..."]
    end
    L1 --> M1
    L2 --> M2
    LB --> MB
    S1 & S2 --> CF["confidence head<br/>P(accept position i)"]
    CF --> AV["adaptive verification:<br/>verify while cumprod(conf) ≥ τ"]
```

* **DFlash**: Markov rank 0, no confidence. Positions are independent given the anchor, so
  acceptance decays along the block.
* **DSpark**: low-rank Markov head (residual softmax regression on the previous token,
  factored by SVD) plus a logistic confidence head. The engine's `confidence_threshold`
  implements adaptive verification.

---

## 8. Training drafters on target traces

Real drafters are trained on **on-policy** data: responses generated by the target, with
the target's hidden states. `speculators` extracts these from vLLM. The lab does the
same thing in miniature:

```mermaid
flowchart LR
    G["generate_corpus(target)<br/>on-policy samples"] --> CT["collect_traces<br/>tokens, [h1;h2] per position,<br/>target probs, copy info"]
    CT --> E1["EagleDrafter.fit<br/>ridge on features (+ TTT)"]
    CT --> B1["BlockDrafter.fit<br/>1. backbone heads (softmax reg.)<br/>2. Markov bias (residual, SVD)<br/>3. confidence (logistic, soft labels)"]
    T["target weights"] --> DM["make_draft_model<br/>perturbed copy = 'smaller sibling'"]
```

---

## 9. Benchmark pipeline (SPEED-Bench style)

```mermaid
flowchart TD
    W["Workloads<br/>realistic: code-edit, rag-qa, multi-turn-chat, open-chat, reasoning<br/>synthetic: random ids, repeated filler"] --> RA
    D["Drafters (built + trained once)"] --> RA
    RA["run_acceptance<br/>lossless generation at k_max<br/>→ SpecStats per (workload, drafter)"] --> TK
    TK["SpecStats.truncated(k)<br/>for every k ≤ k_max"] --> CMX
    CMX["CostModel × batch sizes<br/>tau, verified tokens, DraftCost → tokens/s, speedup"] --> AN
    AN["analyse<br/>• tau per workload<br/>• synthetic vs realistic<br/>• best k per batch size"] --> OUT["speed_bench.md / .json / _rows.csv"]
```

Two-stage by design: **acceptance** depends on the model and the text, not the hardware.
**Throughput** depends on hardware and batch size. Keeping them separate lets one
acceptance run be re-priced for any GPU, model size or batch size.

---

## 10. Cost model

```
forward_ms(B, n) = overhead + max( (weight_bytes + B·ctx·kv_bytes) / HBM_bw ,
                                    2·params·B·n / (peak_flops·mfu) )

speculative step = spec_overhead + draft_ms(B, DraftCost) + forward_ms(B, verified_tokens)
speedup          = (B · tau / step) / (B / forward_ms(B, 1))
```

```
 time per
 target pass      memory-bound                 │ compute-bound
     ▲            (verifying k+1 ≈ verifying 1)│ (each extra token costs FLOPs)
     │                                         │            ╱ n = k+1 tokens
     │                                         │          ╱
     │                                         │        ╱
     │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━╱━━━━━ n = 1 token
     │                                         │    ╱
     └─────────────────────────────────────────┴──────────────────► batch size
              speculation ≈ free here                long drafts waste compute here
```

This is why the best `k` falls as batch grows (`tests/test_cost_model.py::test_best_k_shrinks_with_batch`),
and why speedups quoted at concurrency 1 shrink at higher concurrency. KV-cache reads are
per step, not per token, so at large batch and long context speculation also amortises
KV traffic, which partially offsets the compute penalty.

Profiles: `HARDWARE` (H100, A100, L40S, B200) and `MODELS` (8B, 70B, 30B-A3B MoE) in
`cost_model.py`. Use `specdec whatif` to explore them.

---

## 11. Production workflow with vLLM

```mermaid
sequenceDiagram
    autonumber
    actor U as You
    participant C as specdec replay
    participant B as vLLM baseline
    participant S as vLLM + speculation
    U->>B: scripts/serve.sh TARGET none 8000
    U->>S: scripts/serve.sh TARGET configs/ngram.json 8001
    U->>C: replay --traffic my_traffic.jsonl --concurrency 1,4,16,64
    loop each endpoint × concurrency
        C->>S: GET /metrics (snapshot before)
        par closed-loop workers
            C->>S: POST /v1/chat/completions (stream, include_usage)
            S-->>C: SSE chunks (may carry several accepted tokens)
        end
        C->>S: GET /metrics (snapshot after)
        C->>C: TTFT, TPOT, tok/s, tau = 1 + Δaccepted / Δdrafts
    end
    C-->>U: replay.md / replay.json with speedup vs baseline
```

Notes:

* **TPOT, not chunk gaps.** With speculation, one SSE chunk can contain several tokens.
  The client asks for `usage` in the stream and computes
  `TPOT = (latency − TTFT) / (tokens − 1)`.
* **Closed-loop concurrency.** N workers each keep exactly one request in flight, matching
  `--max-concurrency` in vLLM's benchmark tooling.
* **Acceptance on exactly this traffic.** Prometheus counters are diffed around each run.

Training your own drafter (resource #4) slots in at the left: `speculators` produces a
self-describing checkpoint that `vllm serve <checkpoint>` picks up with no extra flags.
See [training-drafters.md](training-drafters.md).

---

## 12. Mock server

`specdec mock-server` wraps the lab engine in an OpenAI-compatible HTTP server so the whole
production workflow can be exercised without a GPU, in CI and in `scripts/local_demo.sh`.

```mermaid
flowchart LR
    REQ["POST /v1/(chat/)completions"] --> ENC["chars → toy token ids"]
    ENC --> ENG["MockEngine<br/>baseline or speculative_stream"]
    ENG --> SLP["sleep(cost model ms at current<br/>in-flight count × time_scale)"]
    SLP --> SSE["SSE chunk per engine step"]
    ENG --> CNT["vllm:spec_decode_* counters"] --> MET["GET /metrics"]
```

---

## 13. Extension points

| To add... | Do this |
|---|---|
| A drafter | Subclass `Drafter`, implement `propose` (return tempered `probs` you actually sampled from, or `None` if deterministic) and `cost`. Register it in `bench/speed_bench.py::build_drafters`. `tests/test_engine.py` gives you the greedy-equivalence check for free. |
| A workload | Add a `Workload` to `bench/workloads.py::WORKLOADS` with a prompt generator, temperature and output length. |
| Hardware / model profile | Add to `HARDWARE` / `MODELS` in `cost_model.py`; it appears in `--hardware` / `--model-size`. |
| A vLLM method | Add to `serving/configs.py::METHODS`, write a builder, add a JSON under `configs/`. |
| A serving backend | Anything OpenAI-compatible works with `replay`. For metrics, extend `prom.py` with that server's counter names. |

---

## 14. Design decisions

* **numpy only.** Every algorithm runs on a laptop and in CI in seconds, and the code reads
  like the math. GPU work belongs in vLLM and speculators, which this repo drives rather
  than reimplements.
* **One verifier for every method.** Drafters only propose. Losslessness is guaranteed in
  one place and tested once, statistically.
* **Copy head in the toy target.** Real LLMs copy from context (induction heads). Without
  this, n-gram drafting would look useless and code-edit/RAG workloads would be unrealistic.
* **Pending-token rule.** Feature-based drafters cannot see the target's state for the token
  that was just sampled, as in a real engine (see §6).
* **Acceptance and throughput are separate stages.** The prefix-truncation estimator re-scores
  one `k_max` run for every smaller `k`. This holds because verifying a prefix of a draft
  depends only on that prefix.
* **Stdlib HTTP, proxy-free.** `http.client` streams SSE line by line and reaches local
  servers directly, with nothing extra to install.
