# Algorithms

The math behind speculative decoding and each drafter in `specdec`, with pointers to the code.

## 1. Speculative sampling

Let the target define `p(x | context)` and a drafter propose `d_1..d_n` with distributions
`q_1..q_n` (each `d_i ~ q_i`, conditioned on the context plus `d_<i`). One target forward
pass over `[pending, d_1..d_n]` gives `p_1..p_(n+1)`.

For `i = 1..n`:

* accept `d_i` with probability `min(1, p_i(d_i) / q_i(d_i))`;
* on the first rejection, emit `x ~ norm(max(p_i − q_i, 0))` and stop.

If every draft is accepted, emit a bonus token `x ~ p_(n+1)`.

**Why it is lossless.** For one position, the probability of emitting token `x` is

```
P(x) = q(x) · min(1, p(x)/q(x))  +  P(reject) · max(p(x) − q(x), 0) / Σ_y max(p(y) − q(y), 0)
     = min(q(x), p(x))           +  (1 − Σ_y min(p(y), q(y))) · max(p(x) − q(x), 0) / (1 − Σ_y min(p(y), q(y)))
     = min(q(x), p(x)) + max(p(x) − q(x), 0)
     = p(x)
```

By induction over positions, the emitted sequence has exactly the target's distribution
(Leviathan et al., 2023; Chen et al., 2023). Implementation: `specdec/verify.py`.
Statistical test: `tests/test_verify.py`.

**Special cases.**

* *Greedy* (temperature 0): accept while `d_i == argmax p_i`; the correction token is
  `argmax p_i`. Output is token-for-token identical to greedy decoding
  (`tests/test_engine.py::test_greedy_output_identical_to_target`).
* *Deterministic drafters* (n-gram): `q_i` is one-hot on `d_i`, so acceptance is `p_i(d_i)`
  and the residual is `p_i` with `d_i` removed.
* *Temperature* is applied to both distributions as `p^(1/T)` (renormalised) before
  verification (`specdec/sampling.py::apply_temperature`).

## 2. Acceptance length and speedup

Per verification step, the number of emitted tokens is `accepted + 1`. Its mean is the
**acceptance length** τ, which is what vLLM reports as `mean_acceptance_length`:

```
τ = 1 + num_accepted_tokens / num_steps            (1 ≤ τ ≤ k + 1)
draft acceptance rate = num_accepted_tokens / num_draft_tokens
```

With an i.i.d. per-token acceptance probability α and draft length k
(`cost_model.expected_acceptance_length`):

```
E[τ] = (1 − α^(k+1)) / (1 − α)
```

Speedup over plain decoding at batch B (`cost_model.CostModel.speedup`):

```
speedup = τ · t_target(B, 1) / ( t_overhead + t_draft(B, k) + t_target(B, k + 1) )
```

Speculation pays when `t_target(B, k+1) ≈ t_target(B, 1)`, which is the memory-bound regime of small
batches, and when drafting is cheap relative to τ. As B grows, `t_target(B, k+1)` rises
toward `(k+1)·t_target(B, 1)`, and the optimal k falls. See
[ARCHITECTURE.md §10](ARCHITECTURE.md#10-cost-model).

## 3. The toy target (`ToyLM`)

```
h1_t = tanh(W1 h1_(t-1) + E[x_t])
h2_t = tanh(W2 h2_(t-1) + V h1_t)
p(x_(t+1)) = (1 − λ_t) · softmax(U h2_t + b) + λ_t · onehot(copy_t)
```

`copy_t` is the token that followed the most recent earlier occurrence of the longest
context suffix (n ≤ 4), and `λ_t = copy_strength · n / 4`. This "induction head" makes the
toy model copy from context the way real LLMs do. Without it, prompt lookup would never
work and code-edit/RAG workloads would be unrealistic. Two layers give EAGLE-3-style
multi-layer feature fusion something to fuse.

## 4. Drafters

### 4.1 N-gram / prompt lookup (`drafters/ngram.py`)

Find the largest `n ∈ [prompt_lookup_min, prompt_lookup_max]` such that the last n tokens
occurred earlier in the context; propose the k tokens that followed the most recent such
occurrence. There are no parameters to learn. Cost is CPU-side matching (vLLM also lists an
`ngram_gpu` variant).

Trade-off of `prompt_lookup_max`: longer n-grams give fewer but more reliable matches.
`examples/01_ngram_prompt_lookup.py` sweeps it.

### 4.2 Draft model (`drafters/draft_model.py`)

A smaller LM from the same family runs autoregressively for k steps with its own KV cache
(`Session.sync` reuses the longest common prefix after rollbacks). Cost: k sequential
passes of the draft model. The toy draft model is the target with every weight perturbed
by relative Gaussian noise, which produces agreement on easy tokens and divergence on hard ones.

### 4.3 EAGLE-style head (`drafters/eagle.py`)

EAGLE drafts at the **feature** level: given the target's feature vector `f_(t-1)` and the
embedding of the next token, a light head predicts `f_t`, and the target's frozen LM head
turns it into draft logits. Feeding predicted features back in yields k drafts.

```
f̂_t = tanh(A · [f_(t-1); E[x_t]; 1])         (A fitted by ridge regression on atanh(f_t))
q   = mix(softmax(U · top(f̂_t) + b), copy over drafted context)
```

* `layers="top"` uses only the top layer (EAGLE-1 style).
* `layers="all"` fuses `[h1; h2]` (EAGLE-3 uses low/mid/high layers).
* `ttt_rounds` emulates **training-time test**: the head is refitted on inputs produced
  by its own multi-step rollouts, so it learns to correct its own drift deeper into the draft.

Simplifications vs real EAGLE-3: a single linear+tanh layer instead of a transformer
decoder layer, feature regression instead of direct token-level loss, and no draft tree.

### 4.4 DFlash / DSpark-style block drafting (`drafters/block.py`)

**Backbone (DFlash).** One pass maps `[f_(t-1); E[anchor]; 1]` to logits for all B positions
of the block (one softmax-regression head per position). There are no dependencies inside
the block, so later positions must "guess" the earlier ones, and acceptance decays along the block.

**Markov head (DSpark).** A low-rank logit bias from the previous in-block token:

```
logits_i = backbone_i + A[d_(i-1)] · B         (A: V×r, B: r×V; d_0 = anchor)
```

It is fitted as a residual softmax regression (backbone logits as fixed offsets, previous
token one-hot as input), then factored to rank r with a truncated SVD. Sampling is
sequential, but each step is a table lookup plus a vector add: the "sequential
correction" is nearly free.

**Confidence head (DSpark).** A logistic regression on `[max q_i, entropy(q_i), onehot(i), 1]`
predicts `P(position i accepted | earlier positions accepted)`. Training labels are soft:
the target's probability of the drafted token under teacher forcing.

**Adaptive verification.** With `confidence_threshold = τ_c`, the engine verifies only the
prefix where `Π_{j≤i} conf_j ≥ τ_c`. Fewer verified tokens means less compute at large
batch, for a small loss of τ (`examples/05_dspark_block_drafting.py`).

**Copy signal.** Block drafters also get the parallel span copy: for offset i, the token
`i − 1` positions after the source of the context's suffix match. This can be computed from
context alone, so it does not break block parallelism.

## 5. Fitting utilities (`fit.py`)

| Function | Used for |
|---|---|
| `ridge(X, Y, l2)` | EAGLE feature regression (closed form) |
| `fit_softmax_regression(X, y, V, offset=...)` | block backbone heads, Markov residual |
| `fit_logistic_regression(X, y_soft)` | DSpark confidence head |
| `low_rank(M, r)` | Markov head factorisation |

## 6. The k-truncation estimator

Verification of a draft prefix depends only on that prefix. So a step that drafted `d`
tokens and accepted `a` would, with a max draft length `k < d`, have drafted `min(d, k)`
and accepted `min(a, k)`. `SpecStats.truncated(k)` applies this to the per-step log, so one
run at `k_max` prices every smaller k. The contexts visited differ slightly between runs
(trajectories diverge), but the output distribution is identical (lossless), so this is
a sound estimator of per-step statistics.

## References

* Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* 2023.
* Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
* Li et al. *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* 2024 (and EAGLE-2/3).
* Cheng et al. *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation.* 2026.
* vLLM speculative decoding docs; vllm-project/speculators docs.
