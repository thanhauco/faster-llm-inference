"""specdec: speculative decoding, end to end.

Reference implementations of the drafting methods discussed in
"5 resources for faster LLM inference" (n-gram prompt lookup, draft models,
EAGLE-style feature heads, DFlash/DSpark-style block drafting), a lossless
verifier, a SPEED-Bench-style benchmark harness with a roofline cost model,
and tooling to configure, serve and measure speculative decoding in vLLM.
"""

from specdec.engine import GenerationResult, SpecStats, autoregressive_generate, speculative_generate
from specdec.models import ToyLM, make_draft_model
from specdec.verify import verify_draft

__all__ = [
    "GenerationResult",
    "SpecStats",
    "ToyLM",
    "autoregressive_generate",
    "make_draft_model",
    "speculative_generate",
    "verify_draft",
]

__version__ = "0.1.0"
