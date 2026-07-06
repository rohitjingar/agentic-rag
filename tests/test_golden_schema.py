"""The golden set is data, so it gets tested like data."""

from pathlib import Path

import pytest

from eval.schema import GoldenItem, load_golden

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden" / "golden.jsonl"


def test_golden_set_loads_and_is_well_formed():
    items = load_golden(GOLDEN)
    assert len(items) >= 50
    answerable = [i for i in items if i.answerable]
    negatives = [i for i in items if not i.answerable]
    assert len(answerable) >= 46
    assert len(negatives) >= 5  # ~15% negative controls
    # every answerable question has at least one primary label
    for item in answerable:
        assert item.primary_labels, f"{item.qid} has no primary label"


def test_qtype_coverage():
    items = load_golden(GOLDEN)
    present = {i.qtype for i in items}
    assert {"factoid", "how-to", "multi-hop", "vocab-mismatch", "negative-control"} <= present


def test_answerable_requires_gold():
    with pytest.raises(ValueError, match="needs at least one gold"):
        GoldenItem(qid="x", question="q?", expected_answer="a", qtype="factoid", answerable=True)


def test_negative_control_rejects_gold():
    with pytest.raises(ValueError, match="no gold labels"):
        GoldenItem(
            qid="x",
            question="q?",
            expected_answer="a",
            qtype="negative-control",
            answerable=False,
            gold=[{"doc_id": "d", "locator": "x", "grade": "primary"}],
        )
