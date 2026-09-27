import psycopg

from app.core.database import get_connection


def test_get_connection_returns_psycopg_connection():
    connection = get_connection()

    try:
        assert isinstance(connection, psycopg.Connection)
    finally:
        connection.close()


def test_get_connection_connects_to_test_database():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            database_name = cursor.fetchone()[0]

        assert database_name == "rag_document_qa_test"
    finally:
        connection.close()


def test_get_connection_can_execute_query():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()[0]

        assert result == 1
    finally:
        connection.close()


def test_get_connection_is_open_when_returned():
    connection = get_connection()

    try:
        assert connection.closed is False
    finally:
        connection.close()


def test_connection_can_be_closed():
    connection = get_connection()

    connection.close()

    assert connection.closed is True