from io import BytesIO

from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_connection
from app.storage.chunks import (
    create_document,
    insert_chunk,
)
from app.storage.documents import get_document


client = TestClient(app)


def test_get_documents():
    response = client.get("/documents")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_documents_returns_created_document():
    document_id = create_document(
        filename="test.pdf",
        file_size=1024,
        file_path="documents/test.pdf",
    )

    response = client.get("/documents")

    assert response.status_code == 200

    documents = response.json()

    assert len(documents) == 1
    assert documents[0]["id"] == document_id
    assert documents[0]["filename"] == "test.pdf"
    assert documents[0]["file_size"] == 1024
    assert documents[0]["file_path"] == "documents/test.pdf"


def test_upload_invalid_file_type():
    files = {
        "file": (
            "test.txt",
            BytesIO(b"This is not a PDF"),
            "text/plain"
        )
    }

    response = client.post(
        "/documents/upload",
        files=files
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"


def test_upload_missing_filename():
    files = {
        "file": (
            "",
            BytesIO(b"test"),
            "application/pdf"
        )
    }

    response = client.post(
        "/documents/upload",
        files=files
    )

    assert response.status_code in [400, 422]


def test_ask_empty_question():
    response = client.post(
        "/documents/ask",
        json={
            "question": ""
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"


def test_ask_whitespace_question():
    response = client.post(
        "/documents/ask",
        json={
            "question": "   "
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty"


def test_ask_with_empty_document_ids():
    response = client.post(
        "/documents/ask",
        json={
            "question": "What is RAG?",
            "document_ids": []
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "document_ids cannot be empty"


def test_ask_with_nonexistent_document():
    response = client.post(
        "/documents/ask",
        json={
            "question": "What is RAG?",
            "document_ids": [999999]
        }
    )

    assert response.status_code == 404
    assert "Document 999999 not found" in response.json()["detail"]


def test_ask_with_nonexistent_session():
    response = client.post(
        "/documents/ask",
        json={
            "question": "What is RAG?",
            "session_id": 999999
        }
    )

    assert response.status_code == 404
    assert "Chat session 999999 not found" in response.json()["detail"]


def test_document_status_for_nonexistent_document():
    response = client.get(
        "/documents/999999/status"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_document_status_returns_created_document():
    document_id = create_document(
        filename="status-test.pdf",
        file_size=2048,
        file_path="documents/status-test.pdf",
    )

    from app.storage.chunks import update_document_metadata

    update_document_metadata(
        document_id=document_id,
        page_count=5,
        chunk_count=10,
        status="completed",
        stage="completed",
    )

    response = client.get(
        f"/documents/{document_id}/status"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == document_id
    assert data["filename"] == "status-test.pdf"
    assert data["file_size"] == 2048
    assert data["page_count"] == 5
    assert data["chunk_count"] == 10
    assert data["status"] == "completed"
    assert data["stage"] == "completed"


def test_get_pdf_for_nonexistent_document():
    response = client.get(
        "/documents/999999/pdf"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_get_pdf_returns_pdf_file(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    pdf_content = b"%PDF-1.4\nTest PDF content"
    pdf_path.write_bytes(pdf_content)

    document_id = create_document(
        filename="test.pdf",
        file_size=len(pdf_content),
        file_path=str(pdf_path),
    )

    response = client.get(
        f"/documents/{document_id}/pdf"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == pdf_content


def test_get_source_chunk_for_nonexistent_chunk():
    response = client.get(
        "/documents/999999/sources/999999"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Source chunk not found."


def test_get_source_chunk_returns_created_chunk():
    document_id = create_document(
        filename="source-test.pdf",
        file_size=1024,
        file_path="documents/source-test.pdf",
    )

    insert_chunk(
        document_id=document_id,
        source="source-test.pdf",
        page=2,
        chunk_index=3,
        content="RAG retrieves relevant information.",
        embedding=[0.1] * 768,
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM document_chunks
                WHERE document_id = %s
                """,
                (document_id,)
            )

            chunk_id = cursor.fetchone()[0]

    finally:
        connection.close()

    response = client.get(
        f"/documents/{document_id}/sources/{chunk_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["chunk_id"] == chunk_id
    assert data["document_id"] == document_id
    assert data["source"] == "source-test.pdf"
    assert data["page"] == 2
    assert data["chunk"] == 3
    assert data["content"] == "RAG retrieves relevant information."
    assert data["filename"] == "source-test.pdf"
    assert data["file_path"] == "documents/source-test.pdf"


def test_delete_nonexistent_document():
    response = client.delete(
        "/documents/999999"
    )

    assert response.status_code == 404
    assert "Document 999999 not found" in response.json()["detail"]


def test_delete_document_successfully():
    document_id = create_document(
        filename="delete-test.pdf",
        file_size=1024,
        file_path="documents/delete-test.pdf",
    )

    response = client.delete(
        f"/documents/{document_id}"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Document deleted successfully"
    assert data["document_id"] == document_id
    assert data["filename"] == "delete-test.pdf"

    assert get_document(document_id) is None