import pytest

from document_assistant.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"B", "C", "F"}

    assert precision_at_k(retrieved, relevant, k=5) == 0.4


def test_precision_at_k_uses_k_as_denominator():
    retrieved = ["A", "B"]
    relevant = {"B"}

    assert precision_at_k(retrieved, relevant, k=5) == 0.2


def test_precision_at_k_returns_zero_without_relevant_results():
    retrieved = ["A", "B"]
    relevant = {"C", "D"}

    assert precision_at_k(retrieved, relevant, k=2) == 0.0


def test_recall_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"B", "C", "F"}

    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)


def test_recall_at_k_returns_zero_without_relevant_results():
    retrieved = ["A", "B"]
    relevant = {"C", "D"}

    assert recall_at_k(retrieved, relevant, k=2) == 0.0


def test_mrr_returns_reciprocal_rank_of_first_relevant_result():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"B", "C"}

    assert mean_reciprocal_rank(retrieved, relevant, k=4) == 0.5


def test_mrr_returns_zero_when_no_relevant_result_is_found():
    retrieved = ["A", "B", "C"]
    relevant = {"D"}

    assert mean_reciprocal_rank(retrieved, relevant, k=3) == 0.0


@pytest.mark.parametrize("k", [0, -1, -5])
def test_metrics_reject_invalid_k(k):
    retrieved = ["A", "B"]
    relevant = {"A"}

    with pytest.raises(ValueError, match="k must be greater than zero"):
        precision_at_k(retrieved, relevant, k)

    with pytest.raises(ValueError, match="k must be greater than zero"):
        recall_at_k(retrieved, relevant, k)

    with pytest.raises(ValueError, match="k must be greater than zero"):
        mean_reciprocal_rank(retrieved, relevant, k)


def test_recall_rejects_empty_relevant_ids():
    with pytest.raises(ValueError, match="relevant_ids cannot be empty"):
        recall_at_k(["A", "B"], set(), k=2)


def test_mrr_rejects_empty_relevant_ids():
    with pytest.raises(ValueError, match="relevant_ids cannot be empty"):
        mean_reciprocal_rank(["A", "B"], set(), k=2)
