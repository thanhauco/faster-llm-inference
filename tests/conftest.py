import pytest

from specdec.drafters import BlockDrafter, DraftModelDrafter, EagleDrafter, NgramDrafter, collect_traces
from specdec.models import ToyLM, generate_corpus, make_draft_model


@pytest.fixture(scope="session")
def target():
    return ToyLM()


@pytest.fixture(scope="session")
def traces(target):
    return collect_traces(target, generate_corpus(target, 40, 64, temperature=1.0, seed=123))


@pytest.fixture(scope="session")
def drafters(target, traces):
    """Every drafter type, trained small so the suite stays fast."""
    return {
        "ngram": NgramDrafter.vllm_default(),
        "draft-model": DraftModelDrafter(make_draft_model(target)),
        "eagle1": EagleDrafter(target, layers="top").fit(traces),
        "eagle3": EagleDrafter(target).fit(traces),
        "dflash": BlockDrafter.dflash(target, block_size=6).fit(traces, epochs=2, max_rows=4000),
        "dspark": BlockDrafter.dspark(target, block_size=6, markov_rank=16).fit(traces, epochs=2, max_rows=4000),
    }
