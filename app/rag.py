from app.retrieval.search import retrieve
from app.retrieval.reranker import rerank
from app.generation.prompts import build_context
from app.generation.llm import (
    generate_answer,
    generate_answer_stream,
)
from app.retrieval.query_correction import correct_query
from app.retrieval.vocabulary import get_document_vocabulary
from app.retrieval.query_rewrite import rewrite_question
from app.retrieval.query_normalization import normalize_query
from app.retrieval.multi_query import multi_query_retrieve


RERANK_THRESHOLD = 1.0
CORRECTION_MARGIN = 2.0
MULTI_QUERY_COUNT = 3
MULTI_QUERY_TOP_K = 5


def improve_query(
    question: str,
    document_ids: list[int] | None = None,
    retrieval_top_k: int = 10,
    final_top_n: int = 5,
):
    vocabulary = get_document_vocabulary(
        document_ids
    )

    corrected_query, corrections = correct_query(
        question,
        vocabulary,
        similarity_threshold=0.90,
        max_edit_distance=2,
    )

    if not corrections:
        return question

    original_results = retrieve(
        question=question,
        top_k=retrieval_top_k,
        max_distance=0.50,
        document_ids=document_ids,
    )

    original_reranked = rerank(
        question=question,
        results=original_results,
        top_n=final_top_n,
    )

    corrected_results = retrieve(
        question=corrected_query,
        top_k=retrieval_top_k,
        max_distance=0.50,
        document_ids=document_ids,
    )

    corrected_reranked = rerank(
        question=corrected_query,
        results=corrected_results,
        top_n=final_top_n,
    )

    original_score = (
        original_reranked[0]["rerank_score"]
        if original_reranked
        else None
    )

    corrected_score = (
        corrected_reranked[0]["rerank_score"]
        if corrected_reranked
        else None
    )

    if original_score is None and corrected_score is not None:
        return corrected_query

    if original_score is None or corrected_score is None:
        return question

    improvement = corrected_score - original_score

    if improvement >= CORRECTION_MARGIN:
        print(
            f"Query correction accepted: "
            f"'{question}' -> '{corrected_query}' "
            f"(improvement={improvement:.4f})"
        )

        return corrected_query

    print(
        f"Query correction rejected: "
        f"'{question}' -> '{corrected_query}' "
        f"(improvement={improvement:.4f})"
    )

    return question


def retrieve_context(
    question: str,
    document_ids: list[int] | None = None,
    retrieval_top_k: int = 10,
    final_top_n: int = 5,
    max_distance: float = 0.50,
    conversation: list[dict] | None = None,
    apply_threshold: bool = True,
):
    if conversation is None:
        conversation = []

    original_question = question

    # --------------------------------------------------
    # 1. QUERY NORMALIZATION
    # --------------------------------------------------

    question = normalize_query(question)

    if question != original_question:
        print("\n" + "=" * 70)
        print("QUERY NORMALIZATION")
        print("=" * 70)
        print("Original:   ", original_question)
        print("Normalized: ", question)

    # --------------------------------------------------
    # 2. CONVERSATION-AWARE QUERY REWRITE
    # --------------------------------------------------

    rewritten_question = rewrite_question(
        question,
        conversation
    )

    if rewritten_question != question:
        print("\n" + "=" * 70)
        print("QUERY REWRITE")
        print("=" * 70)
        print("Original:   ", question)
        print("Rewritten:  ", rewritten_question)

    question = rewritten_question

    # --------------------------------------------------
    # 3. QUERY CORRECTION
    # --------------------------------------------------

    question = improve_query(
        question=question,
        document_ids=document_ids,
        retrieval_top_k=retrieval_top_k,
        final_top_n=final_top_n,
    )

    # --------------------------------------------------
    # 4. MULTI-QUERY RETRIEVAL
    # --------------------------------------------------

    results = multi_query_retrieve(
        question=question,
        num_queries=MULTI_QUERY_COUNT,
        top_k_per_query=MULTI_QUERY_TOP_K,
        max_distance=max_distance,
        document_ids=document_ids
    )

    print("\n" + "=" * 70)
    print("MULTI-QUERY VECTOR SEARCH RESULTS")
    print("=" * 70)

    if not results:
        print("NO VECTOR RESULTS")

        print(
            f"All results were filtered by "
            f"max_distance = {max_distance}"
        )

        return (
            question,
            None,
            [],
            False,
        )

    for index, result in enumerate(
        results,
        start=1
    ):
        print(f"\nResult {index}")

        print(
            f"Document ID: "
            f"{result['document_id']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_index']}"
        )

    # --------------------------------------------------
    # 5. CROSS-ENCODER RERANKING
    # --------------------------------------------------

    reranked_results = rerank(
        question,
        results,
        top_n=final_top_n
    )

    print("\n" + "=" * 70)
    print("RERANKER RESULTS")
    print("=" * 70)

    for index, result in enumerate(
        reranked_results,
        start=1
    ):
        print(f"\nResult {index}")

        print(
            f"Document ID: "
            f"{result['document_id']}"
        )

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Vector distance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"RRF score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_index']}"
        )

    if not reranked_results:
        return (
            question,
            None,
            [],
            False,
        )

    # --------------------------------------------------
    # 6. RERANK THRESHOLD
    # --------------------------------------------------

    best_score = reranked_results[0]["rerank_score"]

    print("\n" + "=" * 70)
    print("THRESHOLD CHECK")
    print("=" * 70)

    print(
        f"Best rerank score: {best_score:.4f}"
    )

    print(
        f"Required rerank score: "
        f"{RERANK_THRESHOLD:.4f}"
    )

    print(
        f"Passes threshold: "
        f"{best_score >= RERANK_THRESHOLD}"
    )

    if apply_threshold and best_score < RERANK_THRESHOLD:
        print("RESULT REJECTED BY RERANKER")

        return (
            question,
            None,
            [],
            False,
        )

    # --------------------------------------------------
    # 7. BUILD CONTEXT
    # --------------------------------------------------

    context = build_context(
        reranked_results
    )

    return (
        question,
        context,
        reranked_results,
        True,
    )


def answer_question(
    question: str,
    document_ids: list[int] | None = None,
    retrieval_top_k: int = 10,
    final_top_n: int = 5,
    max_distance: float = 0.50,
    conversation: list[dict] | None = None
):
    question, context, reranked_results, has_context = retrieve_context(
        question=question,
        document_ids=document_ids,
        retrieval_top_k=retrieval_top_k,
        final_top_n=final_top_n,
        max_distance=max_distance,
        conversation=conversation,
    )

    if not has_context:
        return (
            "I could not find relevant information "
            "in the selected documents.",
            []
        )

    answer = generate_answer(
        question,
        context
    )

    return answer, reranked_results


def answer_question_stream(
    question: str,
    document_ids: list[int] | None = None,
    retrieval_top_k: int = 10,
    final_top_n: int = 5,
    max_distance: float = 0.50,
    conversation: list[dict] | None = None
):
    question, context, reranked_results, has_context = retrieve_context(
        question=question,
        document_ids=document_ids,
        retrieval_top_k=retrieval_top_k,
        final_top_n=final_top_n,
        max_distance=max_distance,
        conversation=conversation,
    )

    if not has_context:
        yield {
            "type": "complete",
            "content": (
                "I could not find relevant information "
                "in the selected documents."
            ),
            "sources": [],
        }
        return

    for chunk in generate_answer_stream(
        question,
        context
    ):
        yield {
            "type": "token",
            "content": chunk,
        }

    yield {
        "type": "sources",
        "sources": reranked_results,
    }

    yield {
        "type": "done",
    }


if __name__ == "__main__":

    document_ids_input = input(
        "Document IDs (comma-separated, empty for all): "
    ).strip()

    if document_ids_input:
        document_ids = [
            int(document_id.strip())
            for document_id in document_ids_input.split(",")
        ]
    else:
        document_ids = None

    question = input(
        "Question: "
    )

    answer, results = answer_question(
        question=question,
        document_ids=document_ids,
        conversation=[]
    )

    print("\nAnswer:")
    print("=" * 70)
    print(answer)

    if results:
        print("\nSources:")
        print("=" * 70)

        for result in results:
            print(
                f"- Document {result['document_id']} "
                f"| {result['source']} "
                f"| Page {result['page']} "
                f"| Chunk {result['chunk_index']}"
            )