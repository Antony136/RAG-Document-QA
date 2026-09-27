import json

from app.core.database import get_connection


DATASET_PATH = "experiments/evaluation/dataset/questions.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_question_structure(question):
    required_fields = {
        "id",
        "document_id",
        "question",
        "expected_answer",
        "relevant_chunks",
        "difficulty",
        "category",
    }

    missing = required_fields - question.keys()

    if missing:
        raise ValueError(
            f"{question.get('id', '<unknown>')} missing fields: {missing}"
        )

    if not question["relevant_chunks"]:
        raise ValueError(
            f"{question['id']} has no relevant chunks."
        )


def validate_chunks(connection, question):
    cursor = connection.cursor()

    for chunk in question["relevant_chunks"]:
        cursor.execute(
            """
            SELECT 1
            FROM document_chunks
            WHERE document_id = %s
              AND page = %s
              AND chunk_index = %s
            LIMIT 1
            """,
            (
                question["document_id"],
                chunk["page"],
                chunk["chunk_index"],
            ),
        )

        result = cursor.fetchone()

        if result is None:
            raise ValueError(
                f"Invalid chunk for {question['id']}: "
                f"document={question['document_id']}, "
                f"page={chunk['page']}, "
                f"chunk={chunk['chunk_index']}"
            )


def main():
    dataset = load_dataset()

    print("=" * 70)
    print("RAG EVALUATION DATASET VALIDATION")
    print("=" * 70)

    print(f"\nQuestions found: {len(dataset)}")

    ids = [question["id"] for question in dataset]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs found.")

    connection = get_connection()

    try:
        for question in dataset:
            validate_question_structure(question)
            validate_chunks(connection, question)
    finally:
        connection.close()

    print("\n✓ All question structures are valid.")
    print("✓ All referenced document chunks exist.")
    print("✓ No duplicate question IDs found.")

    print("\nDataset summary:")

    categories = {}

    for question in dataset:
        category = question["category"]
        categories[category] = categories.get(category, 0) + 1

    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    print("\n" + "=" * 70)
    print("VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()