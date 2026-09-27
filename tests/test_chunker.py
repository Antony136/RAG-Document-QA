import pytest

from app.ingestion.chunker import (
    split_into_sentences,
    chunk_text,
)


def test_split_into_sentences():
    text = (
        "RAG is useful. "
        "It retrieves information. "
        "Then it generates an answer."
    )

    sentences = split_into_sentences(text)

    assert sentences == [
        "RAG is useful.",
        "It retrieves information.",
        "Then it generates an answer.",
    ]


def test_chunk_short_text():
    text = "RAG retrieves relevant information."

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap_sentences=1,
    )

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_long_text():
    text = (
        "RAG retrieves information from external documents. "
        "The retrieved information is passed to the language model. "
        "The language model uses the context to generate an answer. "
        "This can improve the relevance of generated responses. "
        "The quality of the answer depends on the quality of the retrieved context."
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap_sentences=1,
    )

    assert len(chunks) > 1


def test_chunk_overlap():
    text = (
        "Sentence one contains information about retrieval. "
        "Sentence two explains how documents are searched. "
        "Sentence three describes how relevant chunks are selected. "
        "Sentence four explains how reranking improves retrieval quality. "
        "Sentence five describes how context is passed to the language model. "
        "Sentence six explains how the final answer is generated."
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap_sentences=1,
    )

    assert len(chunks) > 1

    # The final sentence of a previous chunk should
    # appear at the beginning of the next chunk.
    for previous, current in zip(chunks, chunks[1:]):
        previous_sentences = split_into_sentences(previous)

        assert previous_sentences

        last_sentence = previous_sentences[-1]

        assert current.startswith(last_sentence)


def test_chunker_rejects_chunk_size_below_minimum():
    text = "RAG retrieves relevant information."

    with pytest.raises(ValueError, match="at least 200"):
        chunk_text(
            text,
            chunk_size=199,
            overlap_sentences=1,
        )


def test_chunker_accepts_minimum_chunk_size():
    text = (
        "RAG retrieves information from documents. "
        "The retrieved information is provided to the model. "
        "The model uses the context to generate an answer."
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap_sentences=0,
    )

    assert chunks
    assert all(chunks)


def test_chunker_rejects_negative_overlap():
    text = "RAG retrieves relevant information."

    with pytest.raises(ValueError, match="cannot be negative"):
        chunk_text(
            text,
            chunk_size=200,
            overlap_sentences=-1,
        )


def test_empty_text():
    chunks = chunk_text(
        "",
        chunk_size=1000,
        overlap_sentences=1,
    )

    assert chunks == []


def test_whitespace_text():
    chunks = chunk_text(
        "   \n\t   ",
        chunk_size=1000,
        overlap_sentences=1,
    )

    assert chunks == []


def test_chunker_does_not_create_empty_chunks():
    text = (
        "RAG retrieves information. "
        "The retrieved information is useful. "
        "The model generates an answer."
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap_sentences=1,
    )

    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_chunker_preserves_sentence_content():
    text = (
        "RAG retrieves information from documents. "
        "The retrieved information is passed to the model. "
        "The model uses the context to generate an answer."
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap_sentences=0,
    )

    combined_text = " ".join(chunks)

    for sentence in split_into_sentences(text):
        assert sentence in combined_text


def test_chunker_preserves_heading_with_following_content():
    text = (
        "Retrieval Augmented Generation\n"
        "RAG retrieves relevant information from external documents. "
        "The retrieved information is then provided to the language model."
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap_sentences=0,
    )

    assert chunks

    assert any(
        "Retrieval Augmented Generation" in chunk
        and "RAG retrieves relevant information" in chunk
        for chunk in chunks
    )


def test_chunker_preserves_list_items():
    text = (
        "RAG Components\n"
        "1. Retriever\n"
        "2. Embedding model\n"
        "3. Language model"
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap_sentences=0,
    )

    combined_text = " ".join(chunks)

    assert "1. Retriever" in combined_text
    assert "2. Embedding model" in combined_text
    assert "3. Language model" in combined_text


def test_chunker_splits_oversized_sentence():
    text = "RAG " + ("retrieves information " * 100)

    chunks = chunk_text(
        text,
        chunk_size=200,
        overlap_sentences=0,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunker_preserves_abbreviations():
    text = (
        "RAG uses retrieval. "
        "For example, e.g. retrieval searches relevant documents. "
        "It then returns relevant chunks."
    )

    sentences = split_into_sentences(text)

    assert sentences == [
        "RAG uses retrieval.",
        "For example, e.g. retrieval searches relevant documents.",
        "It then returns relevant chunks.",
    ]


def test_chunker_output_is_cleaned():
    text = (
        "RAG retrieves information.   "
        "The model uses the retrieved context.\n\n\n"
        "The final answer is generated."
    )

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap_sentences=0,
    )

    assert chunks

    for chunk in chunks:
        assert chunk == chunk.strip()
        assert "\n\n\n" not in chunk