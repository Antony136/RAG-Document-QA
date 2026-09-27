from math import log2


def recall_at_k(
    retrieved_chunks: list[tuple[int, int]],
    relevant_chunks: set[tuple[int, int]],
    k: int,
) -> float:
    """
    Recall@K = relevant retrieved chunks / total relevant chunks.
    """

    if not relevant_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]

    retrieved_relevant = sum(
        1 for chunk in top_k
        if chunk in relevant_chunks
    )

    return retrieved_relevant / len(relevant_chunks)


def precision_at_k(
    retrieved_chunks: list[tuple[int, int]],
    relevant_chunks: set[tuple[int, int]],
    k: int,
) -> float:
    """
    Precision@K = relevant retrieved chunks / K.

    If fewer than K results are available, the denominator is
    the number of actually retrieved results.
    """

    top_k = retrieved_chunks[:k]

    if not top_k:
        return 0.0

    retrieved_relevant = sum(
        1 for chunk in top_k
        if chunk in relevant_chunks
    )

    return retrieved_relevant / len(top_k)


def hit_rate_at_k(
    retrieved_chunks: list[tuple[int, int]],
    relevant_chunks: set[tuple[int, int]],
    k: int,
) -> float:
    """
    Hit Rate@K is 1 if at least one relevant chunk
    appears in the top K results, otherwise 0.
    """

    top_k = retrieved_chunks[:k]

    return float(
        any(chunk in relevant_chunks for chunk in top_k)
    )


def reciprocal_rank(
    retrieved_chunks: list[tuple[int, int]],
    relevant_chunks: set[tuple[int, int]],
) -> float:
    """
    Reciprocal Rank = 1 / rank of the first relevant result.

    Returns 0 when no relevant result is retrieved.
    """

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk in relevant_chunks:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    results: list[tuple[list[tuple[int, int]], set[tuple[int, int]]]]
) -> float:
    """
    MRR = mean reciprocal rank across all questions.
    """

    if not results:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in results
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(
    retrieved_chunks: list[tuple[int, int]],
    relevant_chunks: set[tuple[int, int]],
    k: int,
) -> float:
    """
    Binary-relevance nDCG@K.

    Relevant chunks have relevance 1.
    Non-relevant chunks have relevance 0.
    """

    if not relevant_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]

    dcg = 0.0

    for rank, chunk in enumerate(top_k, start=1):
        if chunk in relevant_chunks:
            dcg += 1.0 / log2(rank + 1)

    ideal_relevant_count = min(k, len(relevant_chunks))

    ideal_dcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def calculate_retrieval_metrics(
    retrieved_chunks,
    relevant_chunks,
    k_values=(1, 3, 5, 10),
):
    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(
            retrieved_chunks,
            relevant_chunks,
            k,
        )

        metrics[f"precision@{k}"] = precision_at_k(
            retrieved_chunks,
            relevant_chunks,
            k,
        )

        metrics[f"hit_rate@{k}"] = hit_rate_at_k(
            retrieved_chunks,
            relevant_chunks,
            k,
        )

        metrics[f"ndcg@{k}"] = ndcg_at_k(
            retrieved_chunks,
            relevant_chunks,
            k,
        )

    metrics["mrr"] = reciprocal_rank(
        retrieved_chunks,
        relevant_chunks,
    )

    metrics["reciprocal_rank"] = reciprocal_rank(
        retrieved_chunks,
        relevant_chunks,
    )

    metrics["mrr"] = metrics["reciprocal_rank"]

    return metrics