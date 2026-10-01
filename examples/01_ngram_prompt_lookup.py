"""Resource #1: start with n-gram (prompt lookup), the simplest method.

Drafts come from matches in your prompt: no draft model, no training. It
shines when the output repeats the input (code edits, RAG, summaries with
quotes) and does little for open-ended chat. Run:

    python examples/01_ngram_prompt_lookup.py
"""

import numpy as np

from specdec import ToyLM, autoregressive_generate, speculative_generate
from specdec.bench.workloads import WORKLOADS
from specdec.drafters import NgramDrafter
from specdec.serving import ngram_config, serve_command

target = ToyLM()
drafter = NgramDrafter.vllm_default()  # prompt_lookup_max=4

print("The post's config: 5 speculative tokens, lookup of 4")
cfg = ngram_config(num_speculative_tokens=5, prompt_lookup_max=4)
print("  ", serve_command("Qwen/Qwen3-8B", cfg), "\n")

print(f"{'workload':<18} {'tau':>5} {'accept':>7}  {'drafted/step':>12}  lossless")
for name in ("code-edit", "rag-qa", "multi-turn-chat", "open-chat", "reasoning"):
    w = WORKLOADS[name]
    taus, rates, dl, same = [], [], [], True
    for i, prompt in enumerate(w.prompts(target, 4, seed=0)):
        r = speculative_generate(target, drafter, prompt, w.max_new_tokens, k=5,
                                 temperature=w.temperature, rng=np.random.default_rng(i))
        taus.append(r.stats.mean_acceptance_length)
        rates.append(r.stats.draft_acceptance_rate)
        dl.append(r.stats.mean_draft_length)
        if w.temperature == 0:
            same &= r.tokens == autoregressive_generate(target, prompt, w.max_new_tokens).tokens
    print(f"{name:<18} {np.mean(taus):5.2f} {np.mean(rates):7.1%}  {np.mean(dl):12.2f}  "
          f"{'yes' if w.temperature == 0 and same else 'sampled'}")

print("\nNote how often n-gram proposes nothing (drafted/step < 5): no match, no draft, no cost.")
print("Sweep prompt_lookup_max to see the precision/recall trade-off:")
w = WORKLOADS["rag-qa"]
for lookup_max in (1, 2, 3, 4, 6):
    d = NgramDrafter(1, lookup_max)
    taus = [speculative_generate(target, d, p, w.max_new_tokens, k=5, temperature=w.temperature,
                                 rng=np.random.default_rng(0)).stats.mean_acceptance_length
            for p in w.prompts(target, 4, seed=0)]
    print(f"  prompt_lookup_max={lookup_max}: tau={np.mean(taus):.2f}")
