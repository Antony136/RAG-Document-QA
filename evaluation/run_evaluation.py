import json
import time
from datetime import datetime
from pathlib import Path

from app.rag import retrieve_context
from evaluation.metrics.retrieval_metrics import (
    calculate_retrieval_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "dataset"
    / "questions.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

K_VALUES = (1, 3, 5, 10)

RETRIEVAL_TOP_K = 10
FINAL_TOP_N = 10
MAX_DISTANCE = 0.50


def load_dataset():
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def chunk_identity(result):
    """
    Create the exact identity of a retrieved chunk.

    A chunk is uniquely identified by:
        document_id + page + chunk_index
    """

    return (
        result["document_id"],
        result["page"],
        result["chunk_index"],
    )


def evaluate_question(question_data):
    question_id = question_data["id"]
    question = question_data["question"]

    relevant_chunks = {
        (
            question_data["document_id"],
            chunk["page"],
            chunk["chunk_index"],
        )
        for chunk in question_data["relevant_chunks"]
    }

    start_time = time.perf_counter()

    try:
        (
            processed_question,
            _context,
            reranked_results,
            _has_context,
        ) = retrieve_context(
            question=question,
            document_ids=[
                question_data["document_id"]
            ],
            retrieval_top_k=RETRIEVAL_TOP_K,
            final_top_n=FINAL_TOP_N,
            max_distance=MAX_DISTANCE,
            conversation=[],
            apply_threshold=False,
        )

        error = None

    except Exception as exc:
        processed_question = question
        reranked_results = []
        error = f"{type(exc).__name__}: {exc}"

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    retrieved_chunks = [
        chunk_identity(result)
        for result in reranked_results
    ]

    metrics = calculate_retrieval_metrics(
        retrieved_chunks=retrieved_chunks,
        relevant_chunks=relevant_chunks,
        k_values=K_VALUES,
    )

    retrieved_details = []

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):
        retrieved_details.append(
            {
                "rank": rank,
                "document_id": result["document_id"],
                "page": result["page"],
                "chunk_index": result["chunk_index"],
                "distance": result.get("distance"),
                "rrf_score": result.get("rrf_score"),
                "rerank_score": result.get(
                    "rerank_score"
                ),
                "content": result.get("content"),
            }
        )

    return {
        "id": question_id,
        "question": question,
        "processed_question": processed_question,
        "document_id": question_data[
            "document_id"
        ],
        "difficulty": question_data.get(
            "difficulty"
        ),
        "category": question_data.get(
            "category"
        ),
        "relevant_chunks": [
            list(chunk)
            for chunk in sorted(relevant_chunks)
        ],
        "retrieved_chunks": [
            list(chunk)
            for chunk in retrieved_chunks
        ],
        "retrieved_details": retrieved_details,
        "metrics": metrics,
        "latency_ms": round(
            elapsed_ms,
            2,
        ),
        "error": error,
    }


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def aggregate_results(results):
    successful_results = [
        result
        for result in results
        if result["error"] is None
    ]

    aggregate = {}

    for k in K_VALUES:
        aggregate[f"recall@{k}"] = average(
            [
                result["metrics"][
                    f"recall@{k}"
                ]
                for result in successful_results
            ]
        )

        aggregate[f"precision@{k}"] = average(
            [
                result["metrics"][
                    f"precision@{k}"
                ]
                for result in successful_results
            ]
        )

        aggregate[f"hit_rate@{k}"] = average(
            [
                result["metrics"][
                    f"hit_rate@{k}"
                ]
                for result in successful_results
            ]
        )

        aggregate[f"ndcg@{k}"] = average(
            [
                result["metrics"][
                    f"ndcg@{k}"
                ]
                for result in successful_results
            ]
        )

    aggregate["mrr"] = average(
        [
            result["metrics"]["mrr"]
            for result in successful_results
        ]
    )

    aggregate["average_latency_ms"] = average(
        [
            result["latency_ms"]
            for result in successful_results
        ]
    )

    aggregate["questions_evaluated"] = len(
        successful_results
    )

    aggregate["questions_failed"] = (
        len(results)
        - len(successful_results)
    )

    return aggregate


def print_question_result(result):
    print(
        f"\n{'-' * 70}"
    )

    print(
        f"{result['id']}: "
        f"{result['question']}"
    )

    if result["error"]:
        print(
            f"ERROR: {result['error']}"
        )
        return

    print(
        f"Latency: "
        f"{result['latency_ms']:.2f} ms"
    )

    print(
        f"Retrieved: "
        f"{len(result['retrieved_chunks'])}"
    )

    print(
        f"Relevant: "
        f"{len(result['relevant_chunks'])}"
    )

    print("\nMetrics:")

    for k in K_VALUES:
        print(
            f"  Recall@{k}:     "
            f"{result['metrics'][f'recall@{k}']:.4f}"
        )

        print(
            f"  Precision@{k}:  "
            f"{result['metrics'][f'precision@{k}']:.4f}"
        )

        print(
            f"  Hit Rate@{k}:   "
            f"{result['metrics'][f'hit_rate@{k}']:.4f}"
        )

        print(
            f"  nDCG@{k}:       "
            f"{result['metrics'][f'ndcg@{k}']:.4f}"
        )

    print(
        f"  MRR:            "
        f"{result['metrics']['mrr']:.4f}"
    )


def print_summary(aggregate):
    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "AGGREGATE RETRIEVAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestions evaluated: "
        f"{aggregate['questions_evaluated']}"
    )

    print(
        f"Questions failed:     "
        f"{aggregate['questions_failed']}"
    )

    print(
        "\nRetrieval Metrics"
    )

    print("-" * 70)

    for k in K_VALUES:
        print(
            f"Recall@{k:<3}     "
            f"{aggregate[f'recall@{k}']:.4f}"
        )

        print(
            f"Precision@{k:<3}  "
            f"{aggregate[f'precision@{k}']:.4f}"
        )

        print(
            f"Hit Rate@{k:<3}   "
            f"{aggregate[f'hit_rate@{k}']:.4f}"
        )

        print(
            f"nDCG@{k:<3}       "
            f"{aggregate[f'ndcg@{k}']:.4f}"
        )

    print(
        f"MRR              "
        f"{aggregate['mrr']:.4f}"
    )

    print(
        f"\nAverage retrieval latency: "
        f"{aggregate['average_latency_ms']:.2f} ms"
    )


def save_results(results, aggregate):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        RESULTS_DIR
        / f"retrieval_evaluation_{timestamp}.json"
    )

    output = {
        "evaluation": {
            "timestamp": datetime.now().isoformat(),
            "dataset": str(
                DATASET_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "questions": len(results),
            "k_values": list(K_VALUES),
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "final_top_n": FINAL_TOP_N,
            "max_distance": MAX_DISTANCE,
            "threshold_applied": False,
        },
        "aggregate": aggregate,
        "questions": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def main():
    print(
        "=" * 70
    )

    print(
        "RAG RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )

    dataset = load_dataset()

    print(
        f"\nDataset: {DATASET_PATH}"
    )

    print(
        f"Questions: {len(dataset)}"
    )

    print(
        "\nStarting evaluation..."
    )

    results = []

    for index, question_data in enumerate(
        dataset,
        start=1,
    ):
        print(
            f"\n[{index}/{len(dataset)}] "
            f"Evaluating "
            f"{question_data['id']}..."
        )

        result = evaluate_question(
            question_data
        )

        results.append(result)

        print_question_result(
            result
        )

    aggregate = aggregate_results(
        results
    )

    print_summary(
        aggregate
    )

    output_path = save_results(
        results,
        aggregate,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RESULTS SAVED"
    )

    print(
        "=" * 70
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()