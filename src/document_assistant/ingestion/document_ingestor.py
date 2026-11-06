from pathlib import Path

from document_assistant.embeddings.embedder import FastEmbedder
from document_assistant.ingestion.chunker import chunk_pages
from document_assistant.ingestion.pdf_loader import extract_text_from_pdf
from document_assistant.retrieval.vector_store import ChromaVectorStore


class DocumentIngestor:
    """Orchestrate PDF extraction, chunking, embedding, and storage."""

    def __init__(
        self,
        embedder: FastEmbedder | None = None,
        vector_store: ChromaVectorStore | None = None,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> None:
        self._embedder = embedder if embedder is not None else FastEmbedder()
        self._vector_store = (
            vector_store if vector_store is not None else ChromaVectorStore()
        )
        self._chunk_size = chunk_size
        self._overlap = overlap

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        document_id: str,
    ) -> dict[str, int]:
        """Process a PDF and store its chunks and embeddings."""

        pages = extract_text_from_pdf(pdf_path)

        chunks = chunk_pages(
            pages,
            chunk_size=self._chunk_size,
            overlap=self._overlap,
        )

        if not chunks:
            raise ValueError("The PDF contains no extractable text.")

        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])

        stored_chunks = self._vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
        )

        return {
            "pages_processed": len(pages),
            "chunks_created": len(chunks),
            "chunks_stored": stored_chunks,
        }
