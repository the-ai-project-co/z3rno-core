"""v0.20.6 — AUTO routing benchmark v2 runner.

Combines v1 (``retrieval_auto_routing.jsonl``, the regression-gate
set) + v2 extensions (``retrieval_auto_routing_v2.jsonl``, harder
edge cases) into a 101-query corpus and runs the AUTO classifier
across one or more LLM models. Emits per-category accuracy + an
overall score to stdout + a JSON report.

This is a **report-only** harness — no hard gate. The regression
gate lives in ``test_auto_routing_eval.py`` and runs only against
the v1 set so a classifier regression on the same queries flags
clearly. v2 accuracy is logged for posture-tracking.

Operator usage::

    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-...  # optional
    uv run python tests/eval/run_auto_routing_v2.py \\
        --models openai/gpt-4o-mini,openai/gpt-4o \\
        --output reports/auto_routing_v2_$(date +%Y%m%d).json

Cost: ~101 LLM calls per model. At gpt-4o-mini pricing that's
roughly $0.01 per full run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import z3rno_core.retrieval.strategies  # noqa: F401 — strategy registration
from z3rno_core.distill.llm_gateway import get_llm_gateway
from z3rno_core.retrieval.strategies.auto import AutoStrategy

HERE = Path(__file__).parent
V1_FILE = HERE / "retrieval_auto_routing.jsonl"
V2_FILE = HERE / "retrieval_auto_routing_v2.jsonl"


def _load_jsonl(p: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not p.is_file():
        return out
    with p.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.append(json.loads(line))
    return out


async def _run_one_model(model: str, queries: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "set OPENAI_API_KEY (or LLM_API_KEY) — the AUTO classifier "
            "needs a real LLM, not a stub"
        )
    gateway = get_llm_gateway(
        provider="litellm",
        model=model,
        api_key=api_key,
        timeout_seconds=10.0,
        max_retries=2,
    )

    per_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    mismatches: list[dict[str, Any]] = []

    for item in queries:
        expected = item["expected"]
        per_category[expected]["total"] += 1
        auto = AutoStrategy()
        chosen = await auto._classify(
            query=item["query"], llm_gateway=gateway
        )
        if chosen == expected:
            per_category[expected]["correct"] += 1
        else:
            mismatches.append(
                {
                    "query": item["query"],
                    "expected": expected,
                    "got": chosen,
                }
            )

    total_correct = sum(c["correct"] for c in per_category.values())
    total = sum(c["total"] for c in per_category.values())
    return {
        "model": model,
        "overall_accuracy": total_correct / max(total, 1),
        "total": total,
        "correct": total_correct,
        "by_category": dict(per_category),
        "mismatches": mismatches,
    }


async def _main(args: argparse.Namespace) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    queries = _load_jsonl(V1_FILE) + _load_jsonl(V2_FILE)
    print(f"Loaded {len(queries)} queries  (v1={len(_load_jsonl(V1_FILE))} v2={len(_load_jsonl(V2_FILE))})")

    runs: list[dict[str, Any]] = []
    for m in models:
        print(f"\n=== {m} ===")
        try:
            result = await _run_one_model(m, queries)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        print(
            f"  overall: {result['correct']}/{result['total']} "
            f"= {result['overall_accuracy']:.1%}"
        )
        for cat, counts in sorted(result["by_category"].items()):
            acc = counts["correct"] / max(counts["total"], 1)
            print(f"    {cat:10s} {counts['correct']}/{counts['total']} = {acc:.1%}")
        runs.append(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"runs": runs}, indent=2))
        print(f"\nReport → {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="openai/gpt-4o-mini",
        help="comma-separated LiteLLM-namespaced models",
    )
    parser.add_argument(
        "--output", default="", help="optional JSON report path"
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
