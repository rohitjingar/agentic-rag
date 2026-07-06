"""Load the corpus from data/corpus into Document models.

Document ids are the extension-less relative path ("fastapi/tutorial/testing"),
stable across re-fetches so golden-set labels can reference them.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rag.models import Document

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def load_corpus(corpus_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(corpus_dir.rglob("*.md")) + sorted(corpus_dir.rglob("*.mdx")):
        relpath = path.relative_to(corpus_dir).as_posix()
        raw = path.read_text(encoding="utf-8")
        text = _FRONTMATTER.sub("", raw).strip()
        if not text:
            continue
        heading = _H1.search(text)
        title = heading.group(1).strip() if heading else path.stem.replace("-", " ")
        doc_id = relpath.rsplit(".", 1)[0]
        documents.append(
            Document(
                id=doc_id,
                source_path=relpath,
                title=title,
                text=text,
                meta={"source": relpath.split("/", 1)[0]},
                content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    return documents
