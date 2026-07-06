"""Fetch the corpus (decision checkpoint (a)): FastAPI + MCP + pgvector + PostgreSQL docs.

One-time script: the fetched markdown is COMMITTED under data/corpus so
ingestion, evals, and CI are hermetic. manifest.json records resolved commit
SHAs and a corpus_version hash for provenance.

Run: uv run python scripts/fetch_corpus.py
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"

FASTAPI_EXCLUDE = {
    "release-notes.md",
    "fastapi-people.md",
    "external-links.md",
    "newsletter.md",
    "management.md",
    "management-tasks.md",
    "contributing.md",
    "help-fastapi.md",
}

# schema.mdx files are machine-generated typedoc HTML dumps (~1.4 MB of markup,
# no prose). They poisoned retrieval in the Phase 1 probe — data hygiene, out.
MCP_EXCLUDE = {"schema.mdx"}

# Curated PostgreSQL doc pages: retrieval-relevant topics with vocabulary that
# differs from how backend engineers phrase questions (good BM25-vs-dense material).
POSTGRES_PAGES = [
    "indexes",
    "indexes-types",
    "indexes-multicolumn",
    "indexes-partial",
    "indexes-expressional",
    "indexes-examine",
    "textsearch-intro",
    "textsearch-tables",
    "textsearch-controls",
    "textsearch-features",
    "textsearch-dictionaries",
    "textsearch-indexes",
    "textsearch-limitations",
    "performance-tips",
    "using-explain",
    "planner-optimizer",
    "runtime-config-query",
    "runtime-config-resource",
    "populate",
    "routine-vacuuming",
    "mvcc-intro",
    "transaction-iso",
    "explicit-locking",
    "wal-intro",
    "wal-reliability",
    "backup-dump",
    "continuous-archiving",
    "sql-createindex",
    "sql-explain",
    "sql-vacuum",
    "sql-analyze",
    "datatype-json",
    "functions-json",
    "arrays",
    "ddl-partitioning",
    "pgtrgm",
]

LICENSES = """\
# Corpus sources & licenses

| source | upstream | license |
|---|---|---|
| fastapi/ | github.com/fastapi/fastapi (docs/en/docs) | MIT |
| mcp/ | github.com/modelcontextprotocol/modelcontextprotocol (docs) | MIT |
| pgvector/ | github.com/pgvector/pgvector (README) | PostgreSQL-style |
| postgres/ | postgresql.org/docs/17 (curated pages, HTML to Markdown) | PostgreSQL License |

Fetched by scripts/fetch_corpus.py; resolved revisions are in manifest.json.
"""


def gh_json(client: httpx.Client, url: str) -> dict:
    resp = client.get(url, headers={"Accept": "application/vnd.github+json"})
    resp.raise_for_status()
    return resp.json()


def resolve_sha(client: httpx.Client, repo: str, ref: str) -> str:
    return gh_json(client, f"https://api.github.com/repos/{repo}/commits/{ref}")["sha"]


def fetch_tarball(client: httpx.Client, repo: str, sha: str) -> tarfile.TarFile:
    resp = client.get(f"https://codeload.github.com/{repo}/tar.gz/{sha}")
    resp.raise_for_status()
    return tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")


def extract_markdown(
    tar: tarfile.TarFile,
    inner_dir: str,
    out_dir: Path,
    *,
    exclude: set[str] = frozenset(),
    suffixes: tuple[str, ...] = (".md", ".mdx"),
) -> int:
    count = 0
    for member in tar.getmembers():
        parts = member.name.split("/", 1)
        if len(parts) < 2 or not parts[1].startswith(inner_dir):
            continue
        rel = parts[1][len(inner_dir) :].lstrip("/")
        if not rel or not rel.endswith(suffixes) or Path(rel).name in exclude:
            continue
        fileobj = tar.extractfile(member)
        if fileobj is None:
            continue
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fileobj.read())
        count += 1
    return count


def fetch_postgres_pages(client: httpx.Client, out_dir: Path) -> int:
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for slug in POSTGRES_PAGES:
        url = f"https://www.postgresql.org/docs/17/{slug}.html"
        resp = client.get(url)
        resp.raise_for_status()
        content = BeautifulSoup(resp.text, "html.parser").select_one("#docContent")
        if content is None:
            print(f"  !! no #docContent in {url}", file=sys.stderr)
            continue
        markdown = markdownify(str(content), heading_style="ATX")
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        (out_dir / f"{slug}.md").write_text(markdown + "\n", encoding="utf-8")
        count += 1
        time.sleep(0.3)
    return count


def write_manifest(sources: dict) -> str:
    hasher = hashlib.sha256()
    file_count = 0
    for path in sorted(CORPUS_DIR.rglob("*.md")) + sorted(CORPUS_DIR.rglob("*.mdx")):
        rel = path.relative_to(CORPUS_DIR).as_posix()
        hasher.update(rel.encode())
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
        file_count += 1
    version = hasher.hexdigest()[:12]
    manifest = {
        "corpus_version": version,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "file_count": file_count,
        "sources": sources,
    }
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return version


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    sources: dict[str, dict] = {}

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        fastapi_tag = gh_json(
            client, "https://api.github.com/repos/fastapi/fastapi/releases/latest"
        )["tag_name"]
        for name, repo, ref, inner, exclude in [
            ("fastapi", "fastapi/fastapi", fastapi_tag, "docs/en/docs", FASTAPI_EXCLUDE),
            (
                "mcp",
                "modelcontextprotocol/modelcontextprotocol",
                "main",
                "docs",
                MCP_EXCLUDE,
            ),
            ("pgvector", "pgvector/pgvector", "master", "", frozenset()),
        ]:
            sha = resolve_sha(client, repo, ref)
            print(f"fetching {name} @ {ref} ({sha[:10]})...")
            shutil.rmtree(CORPUS_DIR / name, ignore_errors=True)  # clean re-fetch
            tar = fetch_tarball(client, repo, sha)
            if name == "pgvector":
                count = extract_markdown(tar, "", CORPUS_DIR / name, suffixes=("README.md",))
            else:
                count = extract_markdown(tar, inner, CORPUS_DIR / name, exclude=exclude)
            sources[name] = {"repo": repo, "ref": ref, "sha": sha, "files": count}
            print(f"  {count} files")

        print("fetching postgres doc pages...")
        shutil.rmtree(CORPUS_DIR / "postgres", ignore_errors=True)
        pg_count = fetch_postgres_pages(client, CORPUS_DIR / "postgres")
        sources["postgres"] = {
            "base": "https://www.postgresql.org/docs/17/",
            "pages": pg_count,
        }
        print(f"  {pg_count} pages")

    (CORPUS_DIR / "LICENSES.md").write_text(LICENSES)
    version = write_manifest(sources)
    print(f"corpus_version={version}")


if __name__ == "__main__":
    main()
