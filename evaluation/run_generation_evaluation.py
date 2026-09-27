import json
import time
from datetime import datetime
from pathlib import Path

from app.generation.llm import generate_answer
from app.rag import retrieve_context
from evaluation.generation.llm_judge import judge_answer
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
    return (
        result["document_id"],
        result["page"],
        result["chunk_index"],
    )


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def evaluate_question(question_data):
    question_id = question_data["id"]
    question = question_data["question"]
    expected_answer = question_data["expected_answer"]

    relevant_chunks = {
        (
            question_data["document_id"],
            chunk["page"],
            chunk["chunk_index"],
        )
        for chunk in question_data["relevant_chunks"]
    }

    # ---------------------------------------------------------
    # 1. RETRIEVAL
    # ---------------------------------------------------------

    retrieval_start = time.perf_counter()

    try:
        (
            processed_question,
            context,
            reranked_results,
            has_context,
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

        retrieval_error = None

    except Exception as exc:
        processed_question = question
        context = None
        reranked_results = []
        has_context = False
        retrieval_error = (
            f"{type(exc).__name__}: {exc}"
        )

    retrieval_latency_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000

    retrieved_chunks = [
        chunk_identity(result)
        for result in reranked_results
    ]

    retrieval_metrics = calculate_retrieval_metrics(
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

    # ---------------------------------------------------------
    # 2. GENERATION
    # ---------------------------------------------------------

    generated_answer = None
    generation_latency_ms = 0.0
    generation_error = None

    if context and has_context:
        generation_start = time.perf_counter()

        try:
            generated_answer = generate_answer(
                question=processed_question,
                context=context,
            )

        except Exception as exc:
            generation_error = (
                f"{type(exc).__name__}: {exc}"
            )

        generation_latency_ms = (
            time.perf_counter() - generation_start
        ) * 1000

    else:
        generation_error = (
            "Generation skipped because no context "
            "was retrieved."
        )

    # ---------------------------------------------------------
    # 3. LLM JUDGE
    # ---------------------------------------------------------

    judgment = None
    judge_latency_ms = 0.0
    judge_error = None

    if (
        generated_answer
        and context
        and retrieval_error is None
        and generation_error is None
    ):
        judge_start = time.perf_counter()

        try:
            judgment = judge_answer(
                question=processed_question,
                expected_answer=expected_answer,
                context=context,
                generated_answer=generated_answer,
            )

        except Exception as exc:
            judge_error = (
                f"{type(exc).__name__}: {exc}"
            )

        judge_latency_ms = (
            time.perf_counter() - judge_start
        ) * 1000

    else:
        judge_error = (
            "Judging skipped because generation "
            "did not produce an answer."
        )

    # ---------------------------------------------------------
    # 4. FINAL RESULT
    # ---------------------------------------------------------

    total_latency_ms = (
        retrieval_latency_ms
        + generation_latency_ms
        + judge_latency_ms
    )

    error = (
        retrieval_error
        or generation_error
        or judge_error
    )

    return {
        "id": question_id,
        "question": question,
        "processed_question": processed_question,
        "expected_answer": expected_answer,
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
        "retrieval_metrics": retrieval_metrics,
        "context": context,
        "generated_answer": generated_answer,
        "judgment": judgment,
        "latency": {
            "retrieval_ms": round(
                retrieval_latency_ms,
                2,
            ),
            "generation_ms": round(
                generation_latency_ms,
                2,
            ),
            "judge_ms": round(
                judge_latency_ms,
                2,
            ),
            "total_ms": round(
                total_latency_ms,
                2,
            ),
        },
        "error": error,
        "errors": {
            "retrieval": retrieval_error,
            "generation": generation_error,
            "judge": judge_error,
        },
    }


def aggregate_results(results):
    successful_results = [
        result
        for result in results
        if result["error"] is None
    ]

    aggregate = {}

    # ---------------------------------------------------------
    # RETRIEVAL METRICS
    # ---------------------------------------------------------

    for k in K_VALUES:
        aggregate[f"recall@{k}"] = average(
            [
                result["retrieval_metrics"][
                    f"recall@{k}"
                ]
                for result in successful_results
            ]
        )

        aggregate[f"precision@{k}"] = average(
            [
                result["retrieval_metrics"][
                    f"precision@{k}"
                ]
                for result in successful_results
            ]
        )

        aggregate[f"hit_rate@{k}"] = average(
            [
                result["retrieval_metrics"][
                    f"hit_rate@{k}"
                ]
                for result in successful_results
            ]
        )

        aggregate[f"ndcg@{k}"] = average(
            [
                result["retrieval_metrics"][
                    f"ndcg@{k}"
                ]
                for result in successful_results
            ]
        )

    aggregate["mrr"] = average(
        [
            result["retrieval_metrics"]["mrr"]
            for result in successful_results
        ]
    )

    # ---------------------------------------------------------
    # GENERATION METRICS
    # ---------------------------------------------------------

    judged_results = [
        result
        for result in results
        if result["judgment"] is not None
    ]

    for metric_name in (
        "correctness",
        "relevance",
        "faithfulness",
        "context_relevance",
    ):
        scores = [
            result["judgment"][metric_name][
                "score"
            ]
            for result in judged_results
        ]

        aggregate[
            f"average_{metric_name}_score"
        ] = average(scores)

        aggregate[
            f"average_{metric_name}_normalized"
        ] = average(scores) / 5.0

    # ---------------------------------------------------------
    # LATENCY
    # ---------------------------------------------------------

    aggregate["average_retrieval_latency_ms"] = (
        average(
            [
                result["latency"][
                    "retrieval_ms"
                ]
                for result in results
            ]
        )
    )

    aggregate["average_generation_latency_ms"] = (
        average(
            [
                result["latency"][
                    "generation_ms"
                ]
                for result in results
                if result["generated_answer"]
                is not None
            ]
        )
    )

    aggregate["average_judge_latency_ms"] = (
        average(
            [
                result["latency"]["judge_ms"]
                for result in results
                if result["judgment"] is not None
            ]
        )
    )

    aggregate["average_total_latency_ms"] = (
        average(
            [
                result["latency"]["total_ms"]
                for result in results
            ]
        )
    )

    # ---------------------------------------------------------
    # COUNTS
    # ---------------------------------------------------------

    aggregate["questions_evaluated"] = len(
        results
    )

    aggregate["questions_with_generated_answers"] = (
        len(
            [
                result
                for result in results
                if result["generated_answer"]
                is not None
            ]
        )
    )

    aggregate["questions_judged"] = len(
        judged_results
    )

    aggregate["questions_failed"] = len(
        [
            result
            for result in results
            if result["error"] is not None
        ]
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

    print(
        f"\nRetrieval latency: "
        f"{result['latency']['retrieval_ms']:.2f} ms"
    )

    print(
        f"Generation latency: "
        f"{result['latency']['generation_ms']:.2f} ms"
    )

    print(
        f"Judge latency: "
        f"{result['latency']['judge_ms']:.2f} ms"
    )

    print(
        f"Total latency: "
        f"{result['latency']['total_ms']:.2f} ms"
    )

    print(
        f"\nRetrieved chunks: "
        f"{len(result['retrieved_chunks'])}"
    )

    print(
        f"Relevant chunks: "
        f"{len(result['relevant_chunks'])}"
    )

    if result["generated_answer"]:
        print("\nGenerated answer:")
        print(result["generated_answer"])

    if result["judgment"]:
        print("\nGeneration evaluation:")

        print(
            f"  Correctness:       "
            f"{result['judgment']['correctness']['score']}/5"
        )

        print(
            f"  Relevance:         "
            f"{result['judgment']['relevance']['score']}/5"
        )

        print(
            f"  Faithfulness:      "
            f"{result['judgment']['faithfulness']['score']}/5"
        )

        print(
            f"  Context relevance: "
            f"{result['judgment']['context_relevance']['score']}/5"
        )


def print_summary(aggregate):
    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "AGGREGATE RAG GENERATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestions evaluated: "
        f"{aggregate['questions_evaluated']}"
    )

    print(
        f"Generated answers:    "
        f"{aggregate['questions_with_generated_answers']}"
    )

    print(
        f"Questions judged:     "
        f"{aggregate['questions_judged']}"
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
        "\nGeneration Metrics"
    )

    print("-" * 70)

    print(
        f"Correctness       "
        f"{aggregate['average_correctness_score']:.2f}/5 "
        f"({aggregate['average_correctness_normalized']:.4f})"
    )

    print(
        f"Relevance         "
        f"{aggregate['average_relevance_score']:.2f}/5 "
        f"({aggregate['average_relevance_normalized']:.4f})"
    )

    print(
        f"Faithfulness      "
        f"{aggregate['average_faithfulness_score']:.2f}/5 "
        f"({aggregate['average_faithfulness_normalized']:.4f})"
    )

    print(
        f"Context relevance "
        f"{aggregate['average_context_relevance_score']:.2f}/5 "
        f"({aggregate['average_context_relevance_normalized']:.4f})"
    )

    print(
        "\nLatency"
    )

    print("-" * 70)

    print(
        f"Average retrieval: "
        f"{aggregate['average_retrieval_latency_ms']:.2f} ms"
    )

    print(
        f"Average generation: "
        f"{aggregate['average_generation_latency_ms']:.2f} ms"
    )

    print(
        f"Average judge:      "
        f"{aggregate['average_judge_latency_ms']:.2f} ms"
    )

    print(
        f"Average total:      "
        f"{aggregate['average_total_latency_ms']:.2f} ms"
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
        / f"generation_evaluation_{timestamp}.json"
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
            "generator_model": "qwen2.5-coder:7b",
            "judge_model": "qwen2.5-coder:7b",
            "judge_temperature": 0.0,
            "judge_scale": "1-5",
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation quality."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions.",
    )

    args = parser.parse_args()

    print(
        "=" * 70
    )

    print(
        "RAG GENERATION EVALUATION"
    )

    print(
        "=" * 70
    )

    dataset = load_dataset()

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than 0."
            )

        dataset = dataset[:args.limit]

    print(
        f"\nDataset: {DATASET_PATH}"
    )

    print(
        f"Questions: {len(dataset)}"
    )

    print(
        "\nGenerator: qwen2.5-coder:7b"
    )

    print(
        "Judge:     qwen2.5-coder:7b"
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
        aggregate
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