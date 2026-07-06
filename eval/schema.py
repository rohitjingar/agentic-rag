"""Golden-set data model.

A golden item anchors its labels to (doc_id, verbatim locator span) in the
SOURCE document, never to a chunk id. `materialize.py` resolves those spans to
chunk ids for whatever chunk config is live, so re-chunking never silently
invalidates the labels — it just re-materializes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

QType = Literal["factoid", "how-to", "multi-hop", "vocab-mismatch", "negative-control"]
Grade = Literal["primary", "supporting"]


class GoldLabel(BaseModel):
    doc_id: str
    # a verbatim substring of the source document that the answer lives in;
    # materialization finds the chunk(s) covering it
    locator: str
    grade: Grade = "primary"


class GoldenItem(BaseModel):
    qid: str
    question: str
    expected_answer: str
    qtype: QType
    answerable: bool = True
    gold: list[GoldLabel] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> GoldenItem:
        if self.answerable and not self.gold:
            raise ValueError(f"{self.qid}: answerable question needs at least one gold label")
        if not self.answerable and self.gold:
            raise ValueError(f"{self.qid}: negative-control must have no gold labels")
        if not self.answerable and self.qtype != "negative-control":
            raise ValueError(f"{self.qid}: unanswerable question must be qtype negative-control")
        return self

    @property
    def primary_labels(self) -> list[GoldLabel]:
        return [g for g in self.gold if g.grade == "primary"]


def load_golden(path: Path) -> list[GoldenItem]:
    items = [GoldenItem.model_validate_json(line) for line in _nonblank_lines(path)]
    qids = [i.qid for i in items]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate qids in golden set")
    return items


def _nonblank_lines(path: Path):
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("//"):
            yield line


def dump_golden(items: list[GoldenItem], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(i.model_dump(), ensure_ascii=False) for i in items) + "\n",
        encoding="utf-8",
    )
