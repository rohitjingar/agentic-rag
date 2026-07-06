"""Metric math verified against hand-computed values. If these drift, every
eval number in the project is suspect — so they are pinned to exact arithmetic.
"""

from math import isnan, log2

from eval.metrics import mean_ignoring_nan, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k():
    retrieved = ["c1", "c2", "c3", "c4"]
    primary = {"c1", "c5"}  # only c1 is retrieved of the two primaries
    assert recall_at_k(retrieved, primary, 5) == 0.5
    assert recall_at_k(retrieved, primary, 1) == 0.5  # c1 at rank 1
    assert recall_at_k(["c9"], primary, 5) == 0.0


def test_recall_undefined_for_no_primary():
    assert isnan(recall_at_k(["c1"], set(), 5))


def test_reciprocal_rank():
    primary = {"c3"}
    assert reciprocal_rank(["c1", "c2", "c3"], primary, 10) == 1 / 3
    assert reciprocal_rank(["c3", "c1"], primary, 10) == 1.0
    assert reciprocal_rank(["c1", "c2"], primary, 10) == 0.0
    # gold present but beyond k -> miss
    assert reciprocal_rank(["c1", "c2", "c3"], primary, 2) == 0.0


def test_ndcg_matches_hand_computation():
    retrieved = ["c1", "c2", "c3"]
    grades = {"c1": "primary", "c3": "supporting", "c5": "primary"}  # c5 not retrieved
    # DCG = 2/log2(2) + 0/log2(3) + 1/log2(4) = 2 + 0 + 0.5 = 2.5
    dcg = 2 / log2(2) + 0 / log2(3) + 1 / log2(4)
    # ideal order [2, 2, 1]: 2/log2(2) + 2/log2(3) + 1/log2(4)
    idcg = 2 / log2(2) + 2 / log2(3) + 1 / log2(4)
    expected = dcg / idcg
    assert abs(ndcg_at_k(retrieved, grades, 10) - expected) < 1e-9
    assert abs(expected - 0.6646) < 1e-3  # independent sanity check on the formula


def test_ndcg_perfect_and_empty():
    grades = {"c1": "primary", "c2": "supporting"}
    assert abs(ndcg_at_k(["c1", "c2"], grades, 10) - 1.0) < 1e-9  # ideal order
    assert isnan(ndcg_at_k(["c1"], {}, 10))


def test_mean_ignoring_nan():
    assert mean_ignoring_nan([1.0, float("nan"), 3.0]) == 2.0
    assert mean_ignoring_nan([float("nan")]) == 0.0
    assert mean_ignoring_nan([]) == 0.0
