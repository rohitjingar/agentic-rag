"""Generation eval: run the real pipeline, then judge every answer.

    uv run python -m eval.judge.runner --label dense-baseline-gen [--limit N]

For each answerable golden question: retrieve -> generate (llama) -> judge
(qwen) on faithfulness/groundedness/relevance. For each negative control:
generate and check the system refused. Writes a JSON stamped with the golden-set
hash + judge prompt version + pipeline config hash (Phase 8 CI verifies these).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from eval.judge.judge import Judge
from eval.run import GOLDEN_PATH, RESULTS_DIR
from eval.schema import GoldenItem, load_golden
from rag.api.app import chunk_config_from
from rag.config import get_settings
from rag.generation.client import OllamaClient
from rag.generation.prompts import REFUSAL
from rag.ingest.embedder import build_embedder
from rag.query import RAGService

REFUSAL_MARK = "does not contain the answer"


@dataclass
class GenResult:
    qid: str
    qtype: str
    answerable: bool
    refused: bool
    faithfulness: int
    groundedness: int
    relevance: int
    answer: str


@dataclass
class Generated:
    item: GoldenItem
    answer: str
    sources: list  # list[RetrievedChunk]
    refused: bool


def golden_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _refused(answer: str) -> bool:
    return REFUSAL_MARK in answer or answer.strip() == REFUSAL.strip()


def _aggregate(results: list[GenResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    negatives = [r for r in results if not r.answerable]

    def avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "answerable": {
            "n": len(answerable),
            "faithfulness": avg([r.faithfulness for r in answerable]),
            "groundedness": avg([r.groundedness for r in answerable]),
            "relevance": avg([r.relevance for r in answerable]),
        },
        "negatives": {
            "n": len(negatives),
            # correct behavior on an out-of-scope question is to refuse
            "refusal_accuracy": avg([1.0 if r.refused else 0.0 for r in negatives]),
            "faithfulness": avg([r.faithfulness for r in negatives]),
        },
    }


async def run_generation_eval(label: str, limit: int | None, provisional: bool) -> dict:
    settings = get_settings()
    items = load_golden(GOLDEN_PATH)
    if limit:
        # keep a couple of negatives in a limited smoke run
        answerable = [i for i in items if i.answerable][:limit]
        negatives = [i for i in items if not i.answerable][:2]
        items = answerable + negatives

    chunk_config = chunk_config_from(settings)
    pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=4, open=False)
    await pool.open()
    gen_llm = OllamaClient(
        settings.ollama_base_url, settings.generation_model, num_ctx=settings.llm_num_ctx
    )
    judge_llm = OllamaClient(
        settings.ollama_base_url, settings.judge_model, num_ctx=settings.llm_num_ctx
    )
    embedder = build_embedder(
        settings.embedding_backend, settings.embedding_model, settings.embedding_dim
    )
    service = RAGService(pool, embedder, gen_llm, chunk_config, settings.top_k)
    judge = Judge(judge_llm)

    # Two passes so Ollama loads each model once instead of swapping per question
    # (llama and qwen don't both stay resident on 16 GB). Pass 1: generate all.
    # Pass 2: judge all.
    try:
        generated: list[Generated] = []
        for item in items:
            result = await service.answer(item.question)
            generated.append(
                Generated(item, result.answer, result.sources, _refused(result.answer))
            )
            print(f"  generated {item.qid} [{item.qtype}]", flush=True)

        results: list[GenResult] = []
        for g in generated:
            scores = await judge.score(g.item.question, g.answer, g.sources)
            results.append(
                GenResult(
                    qid=g.item.qid,
                    qtype=g.item.qtype,
                    answerable=g.item.answerable,
                    refused=g.refused,
                    faithfulness=scores.faithfulness,
                    groundedness=scores.groundedness,
                    relevance=scores.relevance,
                    answer=g.answer,
                )
            )
            print(f"  judged {g.item.qid} [{g.item.qtype}]", flush=True)
    finally:
        await gen_llm.aclose()
        await judge_llm.aclose()
        await pool.close()

    report = {
        "label": label,
        "suite": "generation",
        "provisional": provisional,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "generation_model": settings.generation_model,
        "judge_model": settings.judge_model,
        "judge_prompt_version": judge.prompt_version,
        "golden_hash": golden_hash(GOLDEN_PATH),
        "chunk_config_hash": chunk_config.config_hash,
        "metrics": _aggregate(results),
        "per_question": [asdict(r) for r in results],
    }
    return report


def _write(report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "").replace("-", "")
    out = RESULTS_DIR / f"gen_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    table = RESULTS_DIR / "GENERATION_RESULTS.md"
    if not table.exists():
        table.write_text(
            "# Generation eval results (LLM-as-judge)\n\n"
            "| label | faithfulness | groundedness | relevance "
            "| refusal acc (neg) | judge | provisional |\n"
            "|---|---|---|---|---|---|---|\n"
        )
    a = report["metrics"]["answerable"]
    neg = report["metrics"]["negatives"]
    with table.open("a") as fh:
        fh.write(
            f"| {report['label']} | {a['faithfulness']} | {a['groundedness']} | {a['relevance']} "
            f"| {neg['refusal_accuracy']} | {report['judge_model']} "
            f"| {'yes' if report['provisional'] else 'no'} |\n"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the generation (LLM-as-judge) eval.")
    parser.add_argument("--label", default="dense-baseline-gen")
    parser.add_argument("--limit", type=int, default=None, help="only N answerable (smoke run)")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run_generation_eval(args.label, args.limit, provisional=not args.final))
    out = _write(report)
    a = report["metrics"]["answerable"]
    neg = report["metrics"]["negatives"]
    print(
        f"\n[{report['label']}] judge={report['judge_model']} "
        f"(prompts {report['judge_prompt_version']})"
    )
    print(
        f"  answerable (n={a['n']}): faithfulness={a['faithfulness']} "
        f"groundedness={a['groundedness']} relevance={a['relevance']}"
    )
    print(f"  negatives  (n={neg['n']}): refusal_accuracy={neg['refusal_accuracy']}")
    print(f"  wrote {out.relative_to(RESULTS_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
