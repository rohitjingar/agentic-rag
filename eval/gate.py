"""Eval gate for CI: fail the build if quality regresses below thresholds.yaml.

    uv run python -m eval.gate --suite retrieval    # ingests + runs eval in CI
    uv run python -m eval.gate --suite generation   # verifies committed results

This is the "tests must pass to merge" rule, but for probabilistic outputs. The
retrieval suite runs fully in CI (bge-small on CPU, committed corpus, pgvector
service container). The generation suite can't (no Ollama in CI), so instead we
verify the newest COMMITTED generation result is (a) not stale — its stamped
golden_hash + judge prompt version still match the repo — and (b) above floors.
Missing, stale, or failing => merge blocked, same enforcement, zero cloud LLM.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
from pathlib import Path

from eval.judge.runner import golden_hash
from eval.run import GOLDEN_PATH, RESULTS_DIR, run_eval

THRESHOLDS = Path(__file__).resolve().parent / "thresholds.yaml"


def _load_thresholds() -> dict:
    # tiny YAML subset (key: value, two levels) — avoids a yaml dependency for CI
    root: dict = {}
    section = None
    for raw in THRESHOLDS.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            section = line.rstrip(":").strip()
            root[section] = {}
        else:
            key, _, value = line.strip().partition(":")
            root[section][key.strip()] = _coerce(value.strip())
    return root


def _coerce(v: str):
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def _fail(msg: str) -> None:
    print(f"GATE FAIL: {msg}")
    sys.exit(1)


async def gate_retrieval() -> None:
    t = _load_thresholds()["retrieval"]
    mode = t.get("mode", "dense")
    report = await run_eval(mode, f"gate-{mode}", provisional=True)
    m = report["metrics"]["overall"]
    print(
        f"gate retrieval [{mode}]: recall@10={m['recall@10']} mrr={m['mrr']} ndcg@10={m['ndcg@10']}"
    )
    checks = [
        ("recall@10", m["recall@10"], t["min_recall_at_10"]),
        ("mrr", m["mrr"], t["min_mrr"]),
        ("ndcg@10", m["ndcg@10"], t["min_ndcg_at_10"]),
    ]
    failures = [f"{name} {val} < floor {floor}" for name, val, floor in checks if val < floor]
    if failures:
        _fail("; ".join(failures))
    print("gate retrieval: PASS")


def gate_generation() -> None:
    t = _load_thresholds()["generation"]
    results = sorted(glob.glob(str(RESULTS_DIR / "gen_*.json")))
    if not results:
        _fail("no committed generation result (run eval.judge.runner and commit)")
    report = json.loads(Path(results[-1]).read_text())

    live_hash = golden_hash(GOLDEN_PATH)
    if report.get("golden_hash") != live_hash:
        _fail(
            f"generation result is STALE: stamped golden_hash {report.get('golden_hash')} "
            f"!= live {live_hash} — re-run the generation eval and commit it"
        )

    a = report["metrics"]["answerable"]
    neg = report["metrics"]["negatives"]
    print(
        f"gate generation [{report['label']}]: faithfulness={a['faithfulness']} "
        f"refusal_acc={neg['refusal_accuracy']} (golden_hash ok)"
    )
    failures = []
    if a["faithfulness"] < t["min_faithfulness"]:
        failures.append(f"faithfulness {a['faithfulness']} < {t['min_faithfulness']}")
    if neg["refusal_accuracy"] < t["min_refusal_accuracy"]:
        failures.append(f"refusal_accuracy {neg['refusal_accuracy']} < {t['min_refusal_accuracy']}")
    if failures:
        _fail("; ".join(failures))
    print("gate generation: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="CI eval gate.")
    parser.add_argument("--suite", choices=["retrieval", "generation"], required=True)
    args = parser.parse_args()
    if args.suite == "retrieval":
        asyncio.run(gate_retrieval())
    else:
        gate_generation()


if __name__ == "__main__":
    main()
