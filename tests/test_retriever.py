from unittest.mock import Mock

import pytest

from document_assistant.retrieval.retriever import Retriever
from document_assistant.retrieval.vector_store import RetrievedChunk


@pytest.fixture
def mock_embedder():
    embedder = Mock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def mock_vector_store():
    vector_store = Mock()
    vector_store.search.return_value = [
        RetrievedChunk(
            text="Machine learning uses algorithms.",
            page_number=2,
            chunk_index=1,
            document_id="doc-1",
            distance=0.15,
        )
    ]
    return vector_store


@pytest.fixture
def retriever(mock_embedder, mock_vector_store):
    return Retriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )


def test_retrieve_generates_query_embedding(retriever, mock_embedder):
    retriever.retrieve("What is machine learning?")

    mock_embedder.embed_query.assert_called_once_with("What is machine learning?")


def test_retrieve_searches_vector_store(retriever, mock_vector_store):
    retriever.retrieve("What is machine learning?")

    mock_vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        n_results=5,
        document_id=None,
    )


def test_retrieve_returns_relevant_chunks(retriever):
    results = retriever.retrieve("What is machine learning?")

    assert len(results) == 1
    assert results[0].text == "Machine learning uses algorithms."
    assert results[0].page_number == 2
    assert results[0].document_id == "doc-1"


def test_retrieve_passes_custom_result_count(retriever, mock_vector_store):
    retriever.retrieve("Python", n_results=3)

    assert mock_vector_store.search.call_args.kwargs["n_results"] == 3


def test_retrieve_passes_document_filter(retriever, mock_vector_store):
    retriever.retrieve("Python", document_id="doc-2")

    assert mock_vector_store.search.call_args.kwargs["document_id"] == "doc-2"


def test_retrieve_returns_empty_list_when_no_chunks_found(retriever, mock_vector_store):
    mock_vector_store.search.return_value = []

    results = retriever.retrieve("Unknown topic")

    assert results == []


def test_retrieve_propagates_embedding_errors(mock_embedder, mock_vector_store):
    mock_embedder.embed_query.side_effect = ValueError("Invalid query")

    retriever = Retriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )

    with pytest.raises(ValueError, match="Invalid query"):
        retriever.retrieve("Some question")

    mock_vector_store.search.assert_not_called()


def test_retrieve_propagates_vector_store_errors(mock_embedder, mock_vector_store):
    mock_vector_store.search.side_effect = RuntimeError("Database error")

    retriever = Retriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )

    with pytest.raises(RuntimeError, match="Database error"):
        retriever.retrieve("Some question")
