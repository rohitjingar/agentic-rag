from rag.ingest.chunker import ChunkConfig, chunk_document
from rag.models import Document


def make_doc(text: str) -> Document:
    return Document(
        id="test/doc",
        source_path="test/doc.md",
        title="Test Doc",
        text=text,
        content_sha256="0" * 64,
    )


PARAGRAPH = (
    "FastAPI is a modern web framework for building APIs with Python. "
    "It is based on standard Python type hints and provides automatic "
    "interactive documentation, data validation and serialization. "
)


def test_chunks_roundtrip_char_offsets():
    doc = make_doc(PARAGRAPH * 30)
    config = ChunkConfig(size_tokens=50, overlap_tokens=10)
    chunks = chunk_document(doc, config)
    assert len(chunks) > 3
    for chunk in chunks:
        assert doc.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.token_count <= config.size_tokens


def test_chunks_overlap_and_cover():
    from itertools import pairwise

    doc = make_doc(PARAGRAPH * 30)
    chunks = chunk_document(doc, ChunkConfig(size_tokens=50, overlap_tokens=10))
    for prev, nxt in pairwise(chunks):
        assert nxt.char_start < prev.char_end  # overlapping windows
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end >= len(doc.text.rstrip()) - 1
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_short_doc_is_single_chunk():
    doc = make_doc("pgvector supports HNSW and IVFFlat indexes.")
    chunks = chunk_document(doc, ChunkConfig(size_tokens=400, overlap_tokens=60))
    assert len(chunks) == 1
    assert chunks[0].text == doc.text


def test_config_hash_tracks_parameters():
    base = ChunkConfig()
    assert base.config_hash == ChunkConfig().config_hash
    assert base.config_hash != ChunkConfig(size_tokens=200).config_hash
    assert base.config_hash != ChunkConfig(tokenizer_model="other/model").config_hash
