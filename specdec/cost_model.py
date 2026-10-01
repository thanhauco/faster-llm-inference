"""Roofline cost model: why the best draft length changes with batch size.

A decode step of a dense transformer must stream all weights from HBM once,
no matter how many tokens it processes, and must do ~2 x params FLOPs per
token. So forward time is roughly::

    t(B, n) = overhead + max( (weight_bytes + kv_bytes(B)) / bandwidth,
                               2 * params * B * n / (peak_flops * mfu) )

for ``B`` sequences each processing ``n`` tokens. At small batch the weight
read dominates and verifying ``k + 1`` tokens costs about the same as
decoding 1: speculation is nearly free and longer drafts win. As the batch
grows the step becomes compute-bound, every extra speculated token costs real
FLOPs, rejected drafts become waste, and the optimal ``k`` shrinks. This is
consistent with SPEED-Bench's finding that the best draft length changes with
batch size, and with speedups that shrink as concurrency grows (the post quotes
2.03x at concurrency 1 vs 1.66x at 16 for EAGLE 3.1).

The numbers here are a deliberately simple model, good for building
intuition and ranking configurations, not for predicting your production
throughput. Measure that on your own traffic (``specdec replay``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from specdec.drafters.base import DraftCost


@dataclass(frozen=True)
class Hardware:
    name: str
    mem_bandwidth_gbs: float  # HBM bandwidth, GB/s
    peak_tflops: float  # dense bf16 TFLOP/s
    mfu: float = 0.5  # achievable fraction of peak in decode/verify kernels
    step_overhead_ms: float = 0.25  # launch / scheduling overhead per forward pass


@dataclass(frozen=True)
class TargetModel:
    name: str
    params_b: float  # parameters in billions (active params for MoE)
    bytes_per_param: float = 2.0
    kv_bytes_per_token: float = 131072.0  # e.g. Llama-3-8B bf16: 32 layers * 8 kv heads * 128 dim * 2 * 2 B
    weight_params_b: float | None = None  # total params to stream (MoE); defaults to params_b


HARDWARE = {
    "h100": Hardware("H100 SXM", mem_bandwidth_gbs=3350, peak_tflops=989),
    "a100": Hardware("A100 80GB", mem_bandwidth_gbs=2039, peak_tflops=312),
    "l40s": Hardware("L40S", mem_bandwidth_gbs=864, peak_tflops=362),
    "b200": Hardware("B200", mem_bandwidth_gbs=8000, peak_tflops=2250),
}

MODELS = {
    "8b": TargetModel("8B dense", params_b=8.0),
    "70b": TargetModel("70B dense", params_b=70.0, kv_bytes_per_token=327680.0),
    "moe-30b-a3b": TargetModel("30B-A3B MoE", params_b=3.3, weight_params_b=30.5, kv_bytes_per_token=98304.0),
}


@dataclass(frozen=True)
class CostModel:
    hardware: Hardware = HARDWARE["h100"]
    model: TargetModel = MODELS["8b"]
    context_len: int = 2048
    # Per-step cost speculation adds on top of the two forward passes: rejection
    # sampling, draft/target bookkeeping, extra scheduling. Small but not zero.
    spec_overhead_ms: float = 0.5

    def forward_ms(self, batch: int, tokens_per_seq: int, params_b: float | None = None,
                   include_kv: bool = True) -> float:
        """Time of one forward pass over ``batch x tokens_per_seq`` tokens.

        ``params_b`` overrides the network size (used for draft networks).
        """
        if tokens_per_seq <= 0:
            return 0.0
        if params_b is None:
            params, weights = self.model.params_b, self.model.weight_params_b or self.model.params_b
        else:
            params = weights = params_b
        bytes_moved = weights * 1e9 * self.model.bytes_per_param
        if include_kv:
            bytes_moved += batch * self.context_len * self.model.kv_bytes_per_token
        mem_ms = bytes_moved / (self.hardware.mem_bandwidth_gbs * 1e9) * 1e3
        flops = 2.0 * params * 1e9 * batch * tokens_per_seq
        compute_ms = flops / (self.hardware.peak_tflops * 1e12 * self.hardware.mfu) * 1e3
        return self.hardware.step_overhead_ms + max(mem_ms, compute_ms)

    def target_step_ms(self, batch: int, verified_tokens: float) -> float:
        """Target verification pass. ``verified_tokens`` may be fractional (adaptive drafts)."""
        lo = int(verified_tokens)
        frac = verified_tokens - lo
        t_lo = self.forward_ms(batch, max(lo, 1))
        if frac == 0:
            return t_lo
        return (1 - frac) * t_lo + frac * self.forward_ms(batch, lo + 1)

    def draft_step_ms(self, batch: int, cost: DraftCost) -> float:
        """Drafting cost per verification step. Draft KV reads are folded into rel_params."""
        if cost.passes == 0:
            return cost.cpu_ms
        draft_params = self.model.params_b * cost.rel_params
        per_pass = self.forward_ms(batch, cost.tokens_per_pass, params_b=draft_params, include_kv=False)
        return cost.cpu_ms + cost.passes * per_pass

    def baseline_tokens_per_s(self, batch: int) -> float:
        """Aggregate decode throughput without speculation."""
        return batch / (self.forward_ms(batch, 1) / 1e3)

    def speculative_tokens_per_s(self, batch: int, acceptance_length: float, cost: DraftCost,
                                 verified_tokens: float) -> float:
        step_ms = self.spec_overhead_ms + self.draft_step_ms(batch, cost) + self.target_step_ms(batch, verified_tokens)
        return batch * acceptance_length / (step_ms / 1e3)

    def speedup(self, batch: int, acceptance_length: float, cost: DraftCost, verified_tokens: float) -> float:
        return self.speculative_tokens_per_s(batch, acceptance_length, cost, verified_tokens) / \
            self.baseline_tokens_per_s(batch)

    def describe(self) -> dict:
        return {
            "hardware": asdict(self.hardware),
            "model": asdict(self.model),
            "context_len": self.context_len,
            "spec_overhead_ms": self.spec_overhead_ms,
        }


def expected_acceptance_length(alpha: float, k: int) -> float:
    """Leviathan et al.: E[tokens per step] = (1 - alpha^(k+1)) / (1 - alpha) for i.i.d. acceptance alpha."""
    if alpha >= 1.0:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)
