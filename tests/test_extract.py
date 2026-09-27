from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.ingestion.extract import clean_page_text, extract_pages


def create_test_pdf(pdf_path: Path):
    pdf = canvas.Canvas(str(pdf_path))

    pdf.drawString(
        100,
        750,
        "RAG retrieves relevant information.",
    )
    pdf.showPage()

    pdf.drawString(
        100,
        750,
        "The language model uses the retrieved context.",
    )
    pdf.showPage()

    pdf.save()


def create_blank_page_pdf(pdf_path: Path):
    pdf = canvas.Canvas(str(pdf_path))

    # Page 1 intentionally contains no text.
    pdf.showPage()

    pdf.drawString(
        100,
        750,
        "This is the second page.",
    )
    pdf.showPage()

    pdf.save()


def create_structured_pdf(pdf_path: Path):
    pdf = canvas.Canvas(str(pdf_path))

    lines = [
        "Introduction",
        "RAG retrieves relevant information.",
        "The retriever searches documents.",
        "1. Retrieve relevant chunks",
        "2. Pass context to the language model",
        "Conclusion",
        "The model generates an answer using the retrieved context.",
    ]

    y = 750

    for line in lines:
        pdf.drawString(100, y, line)
        y -= 25

    pdf.showPage()
    pdf.save()


def test_extract_pages_returns_list(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(pdf_path)

    pages = extract_pages(str(pdf_path))

    assert isinstance(pages, list)
    assert len(pages) == 2


def test_extract_pages_contains_page_numbers(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(pdf_path)

    pages = extract_pages(str(pdf_path))

    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2


def test_extract_pages_preserves_page_order(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(pdf_path)

    pages = extract_pages(str(pdf_path))

    assert pages[0]["page"] < pages[1]["page"]
    assert "RAG retrieves" in pages[0]["text"]
    assert "language model" in pages[1]["text"]


def test_extract_pages_contains_text(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(pdf_path)

    pages = extract_pages(str(pdf_path))

    assert "RAG retrieves relevant information." in pages[0]["text"]
    assert "language model uses the retrieved context." in pages[1]["text"]


def test_extract_pages_skips_empty_pages(tmp_path):
    pdf_path = tmp_path / "blank_page.pdf"

    create_blank_page_pdf(pdf_path)

    pages = extract_pages(str(pdf_path))

    assert len(pages) == 1
    assert pages[0]["page"] == 2
    assert pages[0]["text"] == "This is the second page."


def test_clean_page_text_normalizes_whitespace():
    text = (
        "RAG \u00a0 retrieves\t relevant information.\n"
        "\n\n\n"
        "The model \u00a0 uses the context."
    )

    cleaned = clean_page_text(text)

    assert cleaned == (
        "RAG retrieves relevant information.\n"
        "\n"
        "The model uses the context."
    )


def test_clean_page_text_removes_control_characters():
    text = "RAG\x00 retrieves information.\r\nThe model uses context."

    cleaned = clean_page_text(text)

    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "RAG retrieves information." in cleaned
    assert "The model uses context." in cleaned


def test_clean_page_text_unescapes_html():
    text = "RAG & retrieval <test>"

    cleaned = clean_page_text(text)

    assert cleaned == "RAG & retrieval <test>"


def test_clean_page_text_removes_special_invisible_characters():
    text = "RAG\u00a0retrieves\u200brelevant\ufeffinformation."

    cleaned = clean_page_text(text)

    assert "\u00a0" not in cleaned
    assert "\u200b" not in cleaned
    assert "\ufeff" not in cleaned

    assert cleaned == "RAG retrievesrelevantinformation."


def test_clean_page_text_preserves_paragraph_boundaries():
    text = (
        "First paragraph contains retrieval information.\n"
        "\n"
        "Second paragraph contains generation information."
    )

    cleaned = clean_page_text(text)

    assert cleaned == (
        "First paragraph contains retrieval information.\n"
        "\n"
        "Second paragraph contains generation information."
    )


def test_clean_page_text_joins_broken_lines():
    text = (
    "RAG retrieves relevant\n"
    "information from documents.\n"
    "The retrieved context is\n"
    "passed to the language model."
    )

    cleaned = clean_page_text(text)

    assert cleaned == (
        "RAG retrieves relevant information from documents.\n"
        "The retrieved context is passed to the language model."
    )



def test_clean_page_text_preserves_list_items():
    text = (
        "Retrieval methods:\n"
        "- Vector search\n"
        "- Keyword search\n"
        "- Hybrid search"
    )

    cleaned = clean_page_text(text)
    lines = cleaned.splitlines()

    assert lines[0] == "Retrieval methods:"
    assert lines[1] == "- Vector search"
    assert lines[2] == "- Keyword search"
    assert lines[3] == "- Hybrid search"


def test_clean_page_text_preserves_headings():
    text = (
        "Introduction\n"
        "RAG retrieves relevant information.\n"
        "Conclusion\n"
        "The model generates an answer."
    )

    cleaned = clean_page_text(text)
    lines = cleaned.splitlines()

    assert lines[0] == "Introduction"
    assert lines[1] == "RAG retrieves relevant information."
    assert lines[2] == "Conclusion"
    assert lines[3] == "The model generates an answer."


def test_extract_pages_handles_structured_content(tmp_path):
    pdf_path = tmp_path / "structured.pdf"

    create_structured_pdf(pdf_path)

    pages = extract_pages(str(pdf_path))

    assert len(pages) == 1

    text = pages[0]["text"]

    assert "Introduction" in text
    assert "RAG retrieves relevant information." in text
    assert "1. Retrieve relevant chunks" in text
    assert "2. Pass context to the language model" in text
    assert "Conclusion" in text


def test_clean_page_text_returns_empty_for_empty_input():
    assert clean_page_text("") == ""


def test_clean_page_text_returns_empty_for_whitespace():
    assert clean_page_text("   \n\n\t  ") == ""


def test_extract_pages_raises_for_missing_file(tmp_path):
    pdf_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises((FileNotFoundError, OSError)):
        extract_pages(str(pdf_path))


def test_extract_pages_returns_empty_for_blank_pdf(tmp_path):
    pdf_path = tmp_path / "blank.pdf"

    pdf = canvas.Canvas(str(pdf_path))
    pdf.showPage()
    pdf.save()

    pages = extract_pages(str(pdf_path))

    assert pages == []