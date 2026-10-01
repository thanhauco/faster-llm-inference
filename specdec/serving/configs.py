"""Build ``--speculative-config`` JSON and ``vllm serve`` commands for each method.

Keys follow the vLLM speculative decoding docs (``method``, ``model``,
``num_speculative_tokens``, ``prompt_lookup_min``/``prompt_lookup_max``,
``draft_tensor_parallel_size``, ...). Method names and supported keys move
between vLLM releases, so pin your vLLM version and check
``vllm serve --help=speculative-config`` if a key is rejected.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Iterable

METHODS = ("ngram", "suffix", "draft_model", "eagle", "eagle3", "mtp", "dflash", "dspark")
MODEL_REQUIRED = {"draft_model", "eagle", "eagle3", "dflash", "dspark"}


def speculative_config(method: str, num_speculative_tokens: int | None = None, model: str | None = None,
                       **extra: Any) -> dict:
    """Generic builder with light validation. ``extra`` keys are passed through unchanged."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method in MODEL_REQUIRED and not model:
        raise ValueError(f"method {method!r} needs a draft/speculator model")
    if num_speculative_tokens is not None and (int(num_speculative_tokens) != num_speculative_tokens
                                               or num_speculative_tokens < 1):
        raise ValueError("num_speculative_tokens must be a positive integer")
    cfg: dict[str, Any] = {"method": method}
    if model:
        cfg["model"] = model
    if num_speculative_tokens is not None:
        cfg["num_speculative_tokens"] = int(num_speculative_tokens)
    cfg.update({k: v for k, v in extra.items() if v is not None})
    return cfg


def ngram_config(num_speculative_tokens: int = 5, prompt_lookup_max: int = 4,
                 prompt_lookup_min: int | None = None) -> dict:
    """Resource #1: the post's starting point is 5 speculative tokens with a lookup of 4."""
    if prompt_lookup_max < 1 or (prompt_lookup_min is not None and not 1 <= prompt_lookup_min <= prompt_lookup_max):
        raise ValueError("need 1 <= prompt_lookup_min <= prompt_lookup_max")
    return speculative_config("ngram", num_speculative_tokens, prompt_lookup_max=prompt_lookup_max,
                              prompt_lookup_min=prompt_lookup_min)


def suffix_config(num_speculative_tokens: int = 8, **extra: Any) -> dict:
    return speculative_config("suffix", num_speculative_tokens, **extra)


def draft_model_config(model: str, num_speculative_tokens: int = 5, draft_tensor_parallel_size: int | None = None) -> dict:
    return speculative_config("draft_model", num_speculative_tokens, model,
                              draft_tensor_parallel_size=draft_tensor_parallel_size)


def eagle3_config(model: str, num_speculative_tokens: int = 3, draft_tensor_parallel_size: int | None = None) -> dict:
    """Resource #2: EAGLE-3 style head on the target's hidden states."""
    return speculative_config("eagle3", num_speculative_tokens, model,
                              draft_tensor_parallel_size=draft_tensor_parallel_size)


def mtp_config(num_speculative_tokens: int = 1, model: str | None = None) -> dict:
    """Native multi-token-prediction heads (DeepSeek, Qwen3-Next, MiMo, GLM, Gemma 4 assistants...)."""
    return speculative_config("mtp", num_speculative_tokens, model)


def dflash_config(model: str, num_speculative_tokens: int = 8) -> dict:
    return speculative_config("dflash", num_speculative_tokens, model)


def dspark_config(model: str, num_speculative_tokens: int = 8) -> dict:
    """Resource #5: block drafting + sequential (Markov) correction + confidence head."""
    return speculative_config("dspark", num_speculative_tokens, model)


PRESETS: dict[str, tuple[str, dict]] = {
    # name: (target model, speculative config)
    "ngram-post": ("Qwen/Qwen3-8B", ngram_config(5, 4)),
    "draft-qwen3-0.6b": ("Qwen/Qwen3-8B", draft_model_config("Qwen/Qwen3-0.6B", 5)),
    "eagle3-llama3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct",
                           eagle3_config("RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3", 3)),
    "eagle3-qwen3-8b": ("Qwen/Qwen3-8B", eagle3_config("RedHatAI/Qwen3-8B-speculator.eagle3", 3)),
    "dflash-qwen3-8b": ("Qwen/Qwen3-8B", dflash_config("RedHatAI/Qwen3-8B-speculator.dflash", 8)),
    "mtp-mimo-7b": ("XiaomiMiMo/MiMo-7B-Base", mtp_config(1)),
}


def serve_command(target_model: str, spec_config: dict | None = None, port: int = 8000,
                  extra_args: Iterable[str] = ()) -> str:
    """A copy-pasteable ``vllm serve`` command line."""
    parts = ["vllm", "serve", target_model, "--port", str(port)]
    if spec_config:
        parts += ["--speculative-config", json.dumps(spec_config, separators=(",", ":"))]
    parts += list(extra_args)
    return " ".join(shlex.quote(p) for p in parts)


def speculators_serve_command(speculator_model: str, port: int = 8000, extra_args: Iterable[str] = ()) -> str:
    """Resource #4: a speculators checkpoint is self-describing, so a plain ``vllm serve`` works.

    vLLM reads ``speculators_config`` from the checkpoint's config.json, loads the
    verifier named there and enables speculative decoding with its defaults.
    """
    parts = ["vllm", "serve", speculator_model, "--port", str(port), *extra_args]
    return " ".join(shlex.quote(p) for p in parts)


def offline_snippet(target_model: str, spec_config: dict) -> str:
    """Equivalent offline (``vllm.LLM``) Python snippet."""
    return (
        "from vllm import LLM, SamplingParams\n\n"
        f"llm = LLM(model={target_model!r}, speculative_config={spec_config!r})\n"
        "out = llm.generate([\"The future of AI is\"], SamplingParams(temperature=0.0, max_tokens=128))\n"
        "print(out[0].outputs[0].text)\n"
    )
