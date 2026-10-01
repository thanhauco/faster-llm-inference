"""Read vLLM's Prometheus ``/metrics`` and turn spec-decode counters into acceptance stats.

vLLM exports cumulative counters (names as of vLLM v1)::

    vllm:spec_decode_num_drafts                       verification steps with drafts
    vllm:spec_decode_num_draft_tokens                 proposed draft tokens
    vllm:spec_decode_num_accepted_tokens              accepted draft tokens
    vllm:spec_decode_num_accepted_tokens_per_pos{position="i"}

(each usually exposed with a ``_total`` suffix). Snapshot before and after a
load test, diff, and you get the acceptance on *your* traffic:

    mean acceptance length = 1 + accepted / drafts
    draft acceptance rate  = accepted / draft_tokens
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from specdec.serving.http import get_text

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfINFNaN]+)")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

PREFIX = "vllm:spec_decode_"


def parse_prometheus(text: str) -> list[tuple[str, dict[str, str], float]]:
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        try:
            samples.append((name, dict(_LABEL.findall(labels)), float(value)))
        except ValueError:
            continue
    return samples


@dataclass
class SpecDecodeSnapshot:
    num_drafts: float = 0.0
    num_draft_tokens: float = 0.0
    num_accepted_tokens: float = 0.0
    accepted_per_pos: dict[int, float] = field(default_factory=dict)
    found: bool = False

    @classmethod
    def from_text(cls, text: str) -> "SpecDecodeSnapshot":
        snap = cls()
        for name, labels, value in parse_prometheus(text):
            if not name.startswith(PREFIX):
                continue
            key = name[len(PREFIX):]
            if key.endswith("_total"):
                key = key[: -len("_total")]
            if key == "num_drafts":
                snap.num_drafts += value
            elif key == "num_draft_tokens":
                snap.num_draft_tokens += value
            elif key == "num_accepted_tokens":
                snap.num_accepted_tokens += value
            elif key == "num_accepted_tokens_per_pos" and "position" in labels:
                pos = int(labels["position"])
                snap.accepted_per_pos[pos] = snap.accepted_per_pos.get(pos, 0.0) + value
            else:
                continue
            snap.found = True
        return snap

    def __sub__(self, other: "SpecDecodeSnapshot") -> "SpecDecodeSnapshot":
        positions = set(self.accepted_per_pos) | set(other.accepted_per_pos)
        return SpecDecodeSnapshot(
            self.num_drafts - other.num_drafts,
            self.num_draft_tokens - other.num_draft_tokens,
            self.num_accepted_tokens - other.num_accepted_tokens,
            {p: self.accepted_per_pos.get(p, 0.0) - other.accepted_per_pos.get(p, 0.0) for p in sorted(positions)},
            self.found or other.found,
        )

    @property
    def mean_acceptance_length(self) -> float | None:
        return 1.0 + self.num_accepted_tokens / self.num_drafts if self.num_drafts else None

    @property
    def draft_acceptance_rate(self) -> float | None:
        return self.num_accepted_tokens / self.num_draft_tokens if self.num_draft_tokens else None

    def per_position_acceptance(self) -> list[float]:
        if not self.num_drafts:
            return []
        return [self.accepted_per_pos[p] / self.num_drafts for p in sorted(self.accepted_per_pos)]

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "num_drafts": self.num_drafts,
            "num_draft_tokens": self.num_draft_tokens,
            "num_accepted_tokens": self.num_accepted_tokens,
            "mean_acceptance_length": self.mean_acceptance_length,
            "draft_acceptance_rate": self.draft_acceptance_rate,
            "per_position_acceptance": self.per_position_acceptance(),
        }


def metrics_url(base_url: str) -> str:
    """``http://host:8000/v1`` -> ``http://host:8000/metrics``."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + "/metrics"


def scrape(base_url: str, timeout: float = 10.0) -> SpecDecodeSnapshot:
    return SpecDecodeSnapshot.from_text(get_text(metrics_url(base_url), timeout))
