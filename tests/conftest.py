import os

import pytest
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL_TEST = os.getenv("DATABASE_URL_TEST")

if not DATABASE_URL_TEST:
    raise ValueError(
        "DATABASE_URL_TEST is not set in the .env file"
    )


@pytest.fixture(autouse=True)
def use_test_database(monkeypatch):
    monkeypatch.setattr(
        "app.core.database.DATABASE_URL",
        DATABASE_URL_TEST
    )


@pytest.fixture(autouse=True)
def clean_test_database():
    yield

    from app.core.test_database import get_test_connection

    connection = get_test_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                TRUNCATE TABLE
                    chat_messages,
                    chat_sessions,
                    document_chunks,
                    documents
                RESTART IDENTITY CASCADE
                """
            )

        connection.commit()

    finally:
        connection.close()