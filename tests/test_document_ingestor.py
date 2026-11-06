from unittest.mock import Mock

import pytest

from document_assistant.ingestion.chunker import TextChunk
from document_assistant.ingestion.document_ingestor import DocumentIngestor
from document_assistant.ingestion.pdf_loader import PdfPageText


@pytest.fixture
def mock_embedder():
    embedder = Mock()
    embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
    return embedder


@pytest.fixture
def mock_vector_store():
    vector_store = Mock()
    vector_store.add_chunks.return_value = 2
    return vector_store


@pytest.fixture
def ingestor(mock_embedder, mock_vector_store):
    return DocumentIngestor(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
    )


def test_ingest_pdf_processes_and_stores_document(
    ingestor, mock_embedder, mock_vector_store, monkeypatch
):
    pages = [
        PdfPageText(page_number=1, text="First page content"),
        PdfPageText(page_number=2, text="Second page content"),
    ]

    chunks = [
        TextChunk(text="First chunk", page_number=1, chunk_index=1),
        TextChunk(text="Second chunk", page_number=2, chunk_index=2),
    ]

    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.extract_text_from_pdf",
        Mock(return_value=pages),
    )
    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.chunk_pages",
        Mock(return_value=chunks),
    )

    result = ingestor.ingest_pdf("document.pdf", "doc-1")

    assert result == {
        "pages_processed": 2,
        "chunks_created": 2,
        "chunks_stored": 2,
    }

    mock_embedder.embed_documents.assert_called_once_with(
        ["First chunk", "Second chunk"]
    )

    mock_vector_store.add_chunks.assert_called_once_with(
        chunks=chunks,
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        document_id="doc-1",
    )


def test_ingest_pdf_passes_chunking_configuration(
    mock_embedder, mock_vector_store, monkeypatch
):
    chunk_pages_mock = Mock(
        return_value=[TextChunk(text="Content", page_number=1, chunk_index=1)]
    )

    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.extract_text_from_pdf",
        Mock(return_value=[PdfPageText(page_number=1, text="Content")]),
    )
    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.chunk_pages",
        chunk_pages_mock,
    )

    ingestor = DocumentIngestor(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        chunk_size=500,
        overlap=50,
    )

    ingestor.ingest_pdf("document.pdf", "doc-1")

    chunk_pages_mock.assert_called_once_with(
        [PdfPageText(page_number=1, text="Content")],
        chunk_size=500,
        overlap=50,
    )


def test_ingest_pdf_rejects_document_without_text(
    ingestor, mock_embedder, mock_vector_store, monkeypatch
):
    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.extract_text_from_pdf",
        Mock(return_value=[PdfPageText(page_number=1, text="")]),
    )
    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.chunk_pages",
        Mock(return_value=[]),
    )

    with pytest.raises(ValueError, match="no extractable text"):
        ingestor.ingest_pdf("empty.pdf", "doc-1")

    mock_embedder.embed_documents.assert_not_called()
    mock_vector_store.add_chunks.assert_not_called()


def test_ingest_pdf_propagates_extraction_errors(ingestor):
    with pytest.raises(FileNotFoundError):
        ingestor.ingest_pdf("missing.pdf", "doc-1")


def test_ingest_pdf_propagates_embedding_errors(ingestor, mock_embedder, monkeypatch):
    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.extract_text_from_pdf",
        Mock(return_value=[PdfPageText(page_number=1, text="Content")]),
    )
    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.chunk_pages",
        Mock(return_value=[TextChunk(text="Content", page_number=1, chunk_index=1)]),
    )

    mock_embedder.embed_documents.side_effect = RuntimeError("Embedding failed")

    with pytest.raises(RuntimeError, match="Embedding failed"):
        ingestor.ingest_pdf("document.pdf", "doc-1")


def test_ingest_pdf_propagates_storage_errors(ingestor, mock_vector_store, monkeypatch):
    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.extract_text_from_pdf",
        Mock(return_value=[PdfPageText(page_number=1, text="Content")]),
    )
    monkeypatch.setattr(
        "document_assistant.ingestion.document_ingestor.chunk_pages",
        Mock(return_value=[TextChunk(text="Content", page_number=1, chunk_index=1)]),
    )

    mock_vector_store.add_chunks.side_effect = RuntimeError("Storage failed")

    with pytest.raises(RuntimeError, match="Storage failed"):
        ingestor.ingest_pdf("document.pdf", "doc-1")
