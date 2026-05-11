"""AUTO routing accuracy benchmark — Phase C.4 acceptance test.

Runs the labeled query set in ``tests/eval/retrieval_auto_routing.jsonl``
through the AUTO classifier and asserts ≥ 80 % top-1 accuracy.

Gating:
  * Skipped when no LLM API key is configured. The classifier requires
    a real ``LiteLLMGateway`` — stubs can't approximate the routing
    quality the test exists to measure.
  * Skipped in unit-test mode by default. Set
    ``Z3RNO_RUN_EVAL=1`` to opt in; CI runs this in a separate eval
    workflow gated on an org-level secret.

Why a separate gate (instead of just ``DATABASE_URL``): the eval
makes real LLM calls — slow + costs money. Tying it to a dedicated
flag keeps the standard ``pytest`` run fast and free.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

# Side-effect: register strategies (AUTO + every candidate).
import z3rno_core.retrieval.strategies  # noqa: F401
from z3rno_core.distill.llm_gateway import get_llm_gateway
from z3rno_core.retrieval.strategies.auto import AutoStrategy

_EVAL_FILE = Path(__file__).parent / "eval" / "retrieval_auto_routing.jsonl"
_MIN_ACCURACY = 0.80


def _load_eval_set() -> list[dict[str, str]]:
    if not _EVAL_FILE.is_file():
        return []
    out: list[dict[str, str]] = []
    with _EVAL_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
    return out


def _llm_available() -> bool:
    return bool(
        os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )


_SKIP_NO_FLAG = pytest.mark.skipif(
    os.environ.get("Z3RNO_RUN_EVAL") != "1",
    reason="set Z3RNO_RUN_EVAL=1 to run AUTO routing accuracy eval",
)
_SKIP_NO_LLM = pytest.mark.skipif(
    not _llm_available(),
    reason="LLM_API_KEY or OPENAI_API_KEY required for the AUTO eval",
)


@pytest.mark.eval
@_SKIP_NO_FLAG
@_SKIP_NO_LLM
async def test_auto_routing_accuracy_meets_threshold() -> None:
    eval_set = _load_eval_set()
    assert eval_set, f"expected eval set at {_EVAL_FILE}; got empty"

    api_key = os.environ.get("LLM_API_KEY") or os.environ["OPENAI_API_KEY"]
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
    gateway = get_llm_gateway(
        provider="litellm",
        model=model,
        api_key=api_key,
        timeout_seconds=10.0,
        max_retries=2,
    )

    correct = 0
    mismatches: list[dict[str, Any]] = []
    for item in eval_set:
        query = item["query"]
        expected = item["expected"]
        auto = AutoStrategy()
        chosen = await auto._classify(query=query, llm_gateway=gateway)
        if chosen == expected:
            correct += 1
        else:
            mismatches.append(
                {
                    "query": query,
                    "expected": expected,
                    "got": chosen,
                    "reason": auto.classifier_reason[:120],
                }
            )

    total = len(eval_set)
    accuracy = correct / total
    print(
        f"\nAUTO routing accuracy: {correct}/{total} = {accuracy:.1%}\n"
        f"Threshold: {_MIN_ACCURACY:.0%}"
    )
    if mismatches:
        print("\nMismatches (sample):")
        for m in mismatches[:10]:
            print(
                f"  query: {m['query']!r}\n"
                f"    expected: {m['expected']}  got: {m['got']}  "
                f"reason: {m['reason']}"
            )

    assert accuracy >= _MIN_ACCURACY, (
        f"AUTO routing accuracy {accuracy:.1%} < threshold "
        f"{_MIN_ACCURACY:.0%}. {len(mismatches)} mismatches; see stdout."
    )


def test_eval_file_well_formed() -> None:
    """Sanity-check the eval file even when the LLM eval is skipped.

    Catches typos and structural issues in the labeled set independently
    of LLM availability. Runs in every standard pytest invocation
    because it doesn't make LLM calls.
    """
    eval_set = _load_eval_set()
    assert len(eval_set) >= 30, f"expected ≥30 labeled queries, got {len(eval_set)}"

    valid_strategies = {
        "VECTOR", "LEXICAL", "GRAPH", "TRIPLET", "TRACE", "TEMPORAL", "ASK",
    }
    for item in eval_set:
        assert "query" in item
        assert "expected" in item
        assert item["expected"] in valid_strategies, (
            f"unknown strategy in eval set: {item['expected']!r}"
        )
        assert item["query"].strip(), "empty query"


