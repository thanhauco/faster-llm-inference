"""Draft proposers: n-gram, draft model, EAGLE-style head, DFlash/DSpark-style block."""

from specdec.drafters.base import Draft, DraftCost, Drafter
from specdec.drafters.block import BlockDrafter
from specdec.drafters.draft_model import DraftModelDrafter
from specdec.drafters.eagle import EagleDrafter
from specdec.drafters.ngram import NgramDrafter
from specdec.drafters.traces import Trace, collect_traces

__all__ = [
    "BlockDrafter",
    "Draft",
    "DraftCost",
    "DraftModelDrafter",
    "Drafter",
    "EagleDrafter",
    "NgramDrafter",
    "Trace",
    "collect_traces",
]
