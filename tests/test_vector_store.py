from uuid import uuid4

import chromadb
import pytest

from document_assistant.ingestion.chunker import TextChunk
from document_assistant.retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def vector_store():
    client = chromadb.EphemeralClient()
    collection_name = f"test_{uuid4().hex}"

    return ChromaVectorStore(
        client=client,
        collection_name=collection_name,
    )


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(
            text="Python is a programming language.",
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            text="Machine learning uses algorithms.",
            page_number=2,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_add_chunks_stores_documents(vector_store, sample_chunks, sample_embeddings):
    result = vector_store.add_chunks(
        sample_chunks, sample_embeddings, document_id="doc-1"
    )

    assert result == 2
    assert vector_store._collection.count() == 2


def test_search_returns_relevant_chunks(vector_store, sample_chunks, sample_embeddings):
    vector_store.add_chunks(sample_chunks, sample_embeddings, document_id="doc-1")

    results = vector_store.search([1.0, 0.0, 0.0], n_results=1)

    assert len(results) == 1
    assert results[0].text == "Python is a programming language."
    assert results[0].page_number == 1
    assert results[0].document_id == "doc-1"


def test_search_respects_document_filter(
    vector_store, sample_chunks, sample_embeddings
):
    vector_store.add_chunks(
        sample_chunks[:1], sample_embeddings[:1], document_id="doc-1"
    )
    vector_store.add_chunks(
        sample_chunks[1:], sample_embeddings[1:], document_id="doc-2"
    )

    results = vector_store.search(
        [0.0, 1.0, 0.0],
        document_id="doc-1",
    )

    assert len(results) == 1
    assert results[0].document_id == "doc-1"


def test_search_empty_collection_returns_empty_list(vector_store):
    assert vector_store.search([1.0, 0.0, 0.0]) == []


def test_add_chunks_rejects_mismatched_counts(
    vector_store, sample_chunks, sample_embeddings
):
    with pytest.raises(ValueError, match="must match"):
        vector_store.add_chunks(sample_chunks, sample_embeddings[:1])


def test_add_chunks_rejects_blank_document_id(
    vector_store, sample_chunks, sample_embeddings
):
    with pytest.raises(ValueError, match="Document ID cannot be blank"):
        vector_store.add_chunks(
            sample_chunks,
            sample_embeddings,
            document_id=" ",
        )


def test_add_chunks_rejects_duplicate_chunk_ids(vector_store):
    chunks = [
        TextChunk("First text", 1, 0),
        TextChunk("Second text", 2, 0),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        vector_store.add_chunks(
            chunks,
            [[1.0, 0.0], [0.0, 1.0]],
            document_id="doc-1",
        )


def test_add_chunks_rejects_empty_embeddings(vector_store, sample_chunks):
    with pytest.raises(ValueError, match="cannot be empty"):
        vector_store.add_chunks(
            sample_chunks[:1],
            [[]],
            document_id="doc-1",
        )


def test_search_rejects_empty_query_embedding(vector_store):
    with pytest.raises(ValueError, match="Query embedding cannot be empty"):
        vector_store.search([])


def test_search_rejects_invalid_result_count(vector_store):
    with pytest.raises(ValueError, match="greater than zero"):
        vector_store.search([1.0, 0.0], n_results=0)


def test_search_rejects_blank_document_filter(vector_store):
    with pytest.raises(ValueError, match="Document ID cannot be blank"):
        vector_store.search([1.0, 0.0], document_id=" ")
