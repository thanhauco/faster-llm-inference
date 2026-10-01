# Results (toy target)

Generated with `make results` (`specdec bench --prompts 8 --out docs/results`). The run is
deterministic and takes about 25 s on a laptop CPU. Raw data:
[`results/speed_bench.json`](results/speed_bench.json) (summary) and
[`results/speed_bench_rows.csv`](results/speed_bench_rows.csv) (every workload × drafter × k × batch).

> These numbers come from a toy 2-layer RNN target and a roofline cost model. Use them to
> compare methods and see **trends**. Do not read them as production speedups; for those,
> replay your traffic against vLLM ([benchmarking.md](benchmarking.md)).

## Tables
Cost model: 8B dense on H100 SXM, context 2048 tokens. 8 prompts per workload. Acceptance from a lossless run at k=8.

### Mean acceptance length (tokens per target pass) at k=5

| workload | ngram | draft-model | eagle1 | eagle3 | dflash | dspark | dspark-adaptive |
|---|---|---|---|---|---|---|---|
| code-edit | 4.45 | 6.00 | 6.00 | 6.00 | 5.93 | 5.93 | 5.93 |
| rag-qa | 2.37 | 4.11 | 4.26 | 4.83 | 3.71 | 3.55 | 3.40 |
| multi-turn-chat | 2.30 | 3.53 | 3.73 | 3.97 | 2.79 | 3.42 | 3.35 |
| open-chat | 1.46 | 3.42 | 3.11 | 3.84 | 2.34 | 2.46 | 2.37 |
| reasoning | 2.74 | 4.84 | 4.64 | 5.14 | 4.60 | 4.63 | 4.50 |
| synthetic-random | 2.28 | 3.92 | 4.07 | 4.45 | 3.77 | 3.90 | 3.79 |
| synthetic-repeat | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |

### Synthetic vs realistic prompts (k=5)

| drafter | realistic tau | realistic speedup@B=1 | realistic speedup@B=256 | synthetic tau | synthetic speedup@B=1 | synthetic speedup@B=256 |
|---|---|---|---|---|---|---|
| ngram | 2.66 | 2.40x | 2.07x | 4.14 | 3.74x | 2.45x |
| draft-model | 4.38 | 2.55x | 2.03x | 4.96 | 2.89x | 2.30x |
| eagle1 | 4.35 | 2.84x | 2.08x | 5.04 | 3.29x | 2.41x |
| eagle3 | 4.75 | 3.11x | 2.28x | 5.22 | 3.41x | 2.50x |
| dflash | 3.87 | 3.12x | 1.78x | 4.88 | 3.94x | 2.24x |
| dspark | 4.00 | 3.23x | 1.84x | 4.95 | 3.99x | 2.27x |
| dspark-adaptive | 3.91 | 3.15x | 2.28x | 4.89 | 3.95x | 2.55x |

### Best draft length per batch size (realistic workloads)

| drafter | B=1 | B=4 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|
| ngram | k=8 (3.03x) | k=8 (3.05x) | k=8 (3.09x) | k=8 (2.75x) | k=4 (2.10x) |
| draft-model | k=8 (2.88x) | k=8 (2.94x) | k=8 (3.19x) | k=4 (2.83x) | k=2 (2.32x) |
| eagle1 | k=8 (3.38x) | k=8 (3.45x) | k=8 (3.69x) | k=4 (2.96x) | k=2 (2.36x) |
| eagle3 | k=8 (3.69x) | k=8 (3.76x) | k=8 (4.03x) | k=4 (3.24x) | k=2 (2.51x) |
| dflash | k=8 (4.09x) | k=8 (4.12x) | k=8 (4.24x) | k=4 (2.81x) | k=2 (2.13x) |
| dspark | k=8 (4.22x) | k=8 (4.26x) | k=8 (4.39x) | k=4 (2.89x) | k=2 (2.17x) |
| dspark-adaptive | k=8 (3.71x) | k=8 (3.75x) | k=8 (3.86x) | k=6 (3.17x) | k=4 (2.29x) |

Toy-model numbers: use them to compare methods and see trends, not as production estimates.

## How to read them

**1. Synthetic prompts overestimate.** Every drafter has a higher τ on the synthetic
workloads than on the realistic ones. N-gram is the most extreme (4.14 vs 2.66, +56%),
because repeated filler is the best case for prompt lookup. Random token ids are not
neutral either: the target's continuation of noise is easier to draft than real
open-ended text. This matches SPEED-Bench's warning about synthetic inputs.

**2. Workload decides everything for n-gram.** τ goes from 4.45 on code-edit (output ≈
input) to 1.46 on open chat. Prompt lookup is a great default for edit/RAG-style traffic
and almost a no-op for creative chat. The cost is also near zero when it misses, because
no draft means a normal decode step.

**3. The best k shrinks with batch size.** At B=1-16 verification is memory-bound and the
longest draft tried (k=8) wins for every drafter. At B=64 the optimum drops to k=4 for the
model-based drafters (n-gram stays at 8, adaptive DSpark at 6, because their effective drafts
are often shorter anyway). At B=256 it is k=2 for model-based drafters and k=4 for n-gram and
adaptive DSpark. A k tuned for single-user latency is wrong for a loaded server.

**4. Feature-level drafting wins on acceptance.** The EAGLE-3-style head (fused layers +
training-time test) has the highest realistic τ (4.75) and keeps the highest acceptance
deep into the draft. Per-position acceptance on open chat:

| position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| eagle3 | 0.81 | 0.64 | 0.55 | 0.47 | 0.37 | 0.35 | 0.32 | 0.30 |
| dflash | 0.62 | 0.32 | 0.17 | 0.13 | 0.11 | 0.09 | 0.08 | 0.08 |
| dspark | 0.63 | 0.34 | 0.21 | 0.16 | 0.13 | 0.11 | 0.10 | 0.09 |

**5. Block drafting wins on cost at small batch.** DFlash/DSpark draft a whole block in one
pass, so despite a lower τ they post the best B=1 speedups (≈4.1-4.2x vs 3.7x for EAGLE-3
at k=8). The Markov head (DSpark) slows the decay along the block (table above).

**6. Adaptive verification protects throughput at high load.** DSpark with a confidence
threshold verifies fewer tokens per step. It gives up a little τ (3.91 vs 4.00) but holds
2.28x at B=256 where plain DSpark drops to 1.84x at k=5, and its best k stays longer (k=6 at B=64).

**7. Low-entropy text is where the big numbers live.** All model-based drafters reach
τ ≈ 4.5-5.1 on the greedy "reasoning" workload versus ≈ 2.3-3.8 on open chat. That is the
same pattern as the much higher acceptance on math reported for DSpark on Kimi K3.

## Reproduce and vary

```bash
specdec bench --prompts 8 --out results/                    # same as above
specdec bench --hardware a100 --model-size 70b --out results/a100-70b
specdec bench --drafters ngram,eagle3 --workloads code-edit,open-chat --ks 1,2,4,8
python examples/05_dspark_block_drafting.py                 # adaptive verification thresholds
```

Per-position acceptance is "accepted at position i / drafted at position i". With adaptive
verification, later positions are only drafted when the confidence head is sure, so that
column can rise along the block.
