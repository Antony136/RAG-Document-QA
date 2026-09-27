import pytest

from evaluation.metrics.retrieval_metrics import (
    calculate_retrieval_metrics,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@pytest.fixture
def relevant_chunks():
    return {
        (37, 2),
        (37, 3),
    }


def test_recall_at_k(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 2),
        (37, 10),
        (37, 3),
    ]

    assert recall_at_k(retrieved, relevant_chunks, 4) == 1.0


def test_recall_at_k_partial(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 2),
        (37, 10),
    ]

    assert recall_at_k(retrieved, relevant_chunks, 3) == 0.5


def test_precision_at_k(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 2),
        (37, 10),
        (37, 3),
        (37, 6),
    ]

    assert precision_at_k(retrieved, relevant_chunks, 5) == 0.4


def test_precision_with_no_results(relevant_chunks):
    assert precision_at_k([], relevant_chunks, 5) == 0.0


def test_hit_rate_when_relevant_result_exists(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 2),
        (37, 10),
    ]

    assert hit_rate_at_k(retrieved, relevant_chunks, 3) == 1.0


def test_hit_rate_when_no_relevant_result(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 10),
        (37, 6),
    ]

    assert hit_rate_at_k(retrieved, relevant_chunks, 3) == 0.0


def test_reciprocal_rank(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 2),
        (37, 10),
        (37, 3),
    ]

    assert reciprocal_rank(retrieved, relevant_chunks) == 0.5


def test_reciprocal_rank_when_no_match(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 10),
        (37, 6),
    ]

    assert reciprocal_rank(retrieved, relevant_chunks) == 0.0


def test_mean_reciprocal_rank():
    results = [
        (
            [(37, 2), (37, 8)],
            {(37, 2)},
        ),
        (
            [(37, 8), (37, 3)],
            {(37, 3)},
        ),
    ]

    assert mean_reciprocal_rank(results) == 0.75


def test_ndcg_perfect_ranking(relevant_chunks):
    retrieved = [
        (37, 2),
        (37, 3),
    ]

    assert ndcg_at_k(
        retrieved,
        relevant_chunks,
        2,
    ) == pytest.approx(1.0)


def test_ndcg_worse_ranking(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 2),
        (37, 10),
        (37, 3),
    ]

    score = ndcg_at_k(
        retrieved,
        relevant_chunks,
        4,
    )

    assert 0.0 < score < 1.0


def test_calculate_retrieval_metrics(relevant_chunks):
    retrieved = [
        (37, 8),
        (37, 2),
        (37, 10),
        (37, 3),
        (37, 6),
    ]

    metrics = calculate_retrieval_metrics(
        retrieved,
        relevant_chunks,
        k_values=(1, 3, 5),
    )

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@3"] == 0.5
    assert metrics["recall@5"] == 1.0

    assert metrics["precision@1"] == 0.0
    assert metrics["precision@3"] == pytest.approx(1 / 3)
    assert metrics["precision@5"] == 0.4

    assert metrics["hit_rate@1"] == 0.0
    assert metrics["hit_rate@3"] == 1.0
    assert metrics["hit_rate@5"] == 1.0

    assert metrics["reciprocal_rank"] == 0.5