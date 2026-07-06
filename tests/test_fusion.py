from rag.models import RetrievedChunk
from rag.retrieval.fusion import reciprocal_rank_fusion


def chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, doc_id="d", ord=0, text=cid, score=0.0)


def test_rrf_rewards_agreement_across_rankers():
    dense = [chunk("c1"), chunk("c2"), chunk("c3")]
    sparse = [chunk("c3"), chunk("c4"), chunk("c1")]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    # c1 (ranks 1,3) and c3 (ranks 3,1) each appear in both -> tie at the top
    expected_top = 1 / 61 + 1 / 63
    top_ids = {fused[0].id, fused[1].id}
    assert top_ids == {"c1", "c3"}
    assert abs(fused[0].score - expected_top) < 1e-12
    # singletons c2, c4 (one list each) rank below
    assert {fused[2].id, fused[3].id} == {"c2", "c4"}
    assert abs(fused[2].score - 1 / 62) < 1e-12


def test_rrf_pure_topk_truncation():
    dense = [chunk("a"), chunk("b"), chunk("c")]
    fused = reciprocal_rank_fusion([dense], k=60, top_k=2)
    assert [c.id for c in fused] == ["a", "b"]


def test_rrf_single_ranker_preserves_order():
    dense = [chunk("x"), chunk("y"), chunk("z")]
    fused = reciprocal_rank_fusion([dense], k=60)
    assert [c.id for c in fused] == ["x", "y", "z"]
    # score strictly decreasing with rank
    assert fused[0].score > fused[1].score > fused[2].score
