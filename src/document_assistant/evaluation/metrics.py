from collections.abc import Collection, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero.")


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """Calculate the proportion of relevant results among the top k positions."""
    _validate_k(k)

    relevant = set(relevant_ids)
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for item_id in top_k if item_id in relevant)

    return relevant_count / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """Calculate the proportion of relevant items retrieved in the top k."""
    _validate_k(k)

    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError("relevant_ids cannot be empty.")

    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for item_id in top_k if item_id in relevant)

    return relevant_count / len(relevant)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int,
) -> float:
    """Calculate the reciprocal rank of the first relevant result in top k."""
    _validate_k(k)

    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError("relevant_ids cannot be empty.")

    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant:
            return 1 / rank

    return 0.0
