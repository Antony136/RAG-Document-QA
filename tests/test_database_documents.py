from app.core.database import get_connection
from app.storage.chunks import (
    create_document,
    update_document_metadata,
    update_document_status,
    get_document_status,
)
from app.storage.documents import (
    get_documents,
    get_document,
)


def test_create_document():
    filename = "pytest_test_document.pdf"

    document_id = create_document(filename)

    assert document_id is not None
    assert isinstance(document_id, int)

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, filename
                FROM documents
                WHERE id = %s
                """,
                (document_id,),
            )

            row = cursor.fetchone()

        assert row is not None
        assert row[0] == document_id
        assert row[1] == filename

    finally:
        connection.close()


def test_create_document_with_metadata():
    filename = "pytest_metadata_document.pdf"
    file_size = 12345
    file_path = "uploads/pytest_metadata_document.pdf"

    document_id = create_document(
        filename=filename,
        file_size=file_size,
        file_path=file_path,
    )

    document = get_document(document_id)

    assert document is not None
    assert document["filename"] == filename
    assert document["file_size"] == file_size
    assert document["file_path"] == file_path
    assert document["status"] == "processing"


def test_get_documents():
    filename = "pytest_retrieval_document.pdf"

    document_id = create_document(filename)

    documents = get_documents()

    assert isinstance(documents, list)

    matching_documents = [
        document
        for document in documents
        if document["id"] == document_id
    ]

    assert len(matching_documents) == 1

    document = matching_documents[0]

    assert document["filename"] == filename
    assert document["chunk_count"] == 0


def test_get_documents_returns_multiple_documents():
    first_id = create_document("pytest_first_document.pdf")
    second_id = create_document("pytest_second_document.pdf")

    documents = get_documents()

    document_ids = [document["id"] for document in documents]

    assert first_id in document_ids
    assert second_id in document_ids


def test_get_document_returns_existing_document():
    filename = "pytest_get_document.pdf"

    document_id = create_document(filename)

    document = get_document(document_id)

    assert document is not None
    assert document["id"] == document_id
    assert document["filename"] == filename
    assert document["chunk_count"] == 0
    assert document["status"] == "processing"


def test_get_document_returns_none_for_missing_document():
    document = get_document(999999)

    assert document is None


def test_update_document_metadata():
    document_id = create_document(
        filename="pytest_update_metadata.pdf",
    )

    update_document_metadata(
        document_id=document_id,
        page_count=10,
        chunk_count=25,
        status="completed",
        stage="finished",
    )

    document = get_document(document_id)

    assert document is not None
    assert document["page_count"] == 10
    assert document["chunk_count"] == 25
    assert document["status"] == "completed"
    assert document["stage"] == "finished"


def test_update_document_status():
    document_id = create_document(
        filename="pytest_update_status.pdf",
    )

    update_document_status(
        document_id=document_id,
        status="processing",
        stage="embedding",
    )

    document = get_document(document_id)

    assert document is not None
    assert document["status"] == "processing"
    assert document["stage"] == "embedding"


def test_get_document_status_returns_existing_document():
    filename = "pytest_document_status.pdf"

    document_id = create_document(filename)

    status = get_document_status(document_id)

    assert status is not None
    assert status["id"] == document_id
    assert status["filename"] == filename
    assert status["status"] == "processing"


def test_get_document_status_returns_none_for_missing_document():
    status = get_document_status(999999)

    assert status is None