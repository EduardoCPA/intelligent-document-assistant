from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from document_assistant.ingestion.chunker import TextChunk


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A text chunk retrieved from the vector database."""

    text: str
    page_number: int
    chunk_index: int
    document_id: str
    distance: float


class ChromaVectorStore:
    """Store and search document embeddings using ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = "document_chunks",
        client: Any | None = None,
    ) -> None:
        """Initialize a persistent ChromaDB collection."""
        if not collection_name.strip():
            raise ValueError("Collection name cannot be blank.")

        self._client = (
            client
            if client is not None
            else chromadb.PersistentClient(path=str(persist_directory))
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        document_id: str = "default",
    ) -> int:
        """Store chunks and their embeddings in ChromaDB."""
        if not document_id.strip():
            raise ValueError("Document ID cannot be blank.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        ids = [f"{document_id}:{chunk.chunk_index}" for chunk in chunks]

        if len(set(ids)) != len(ids):
            raise ValueError("Chunk IDs must be unique within a document.")

        if any(len(embedding) == 0 for embedding in embeddings):
            raise ValueError("Embeddings cannot be empty.")

        self._collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=[
                {
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "document_id": document_id,
                }
                for chunk in chunks
            ],
        )

        return len(chunks)

    def search(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most similar document chunks."""
        if len(query_embedding) == 0:
            raise ValueError("Query embedding cannot be empty.")

        if n_results <= 0:
            raise ValueError("n_results must be greater than zero.")

        if document_id is not None and not document_id.strip():
            raise ValueError("Document ID cannot be blank.")

        collection_count = self._collection.count()

        if collection_count == 0:
            return []

        query_arguments = {
            "query_embeddings": [list(query_embedding)],
            "n_results": min(n_results, collection_count),
        }

        if document_id is not None:
            query_arguments["where"] = {"document_id": document_id}

        results = self._collection.query(**query_arguments)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if documents is None or metadatas is None or distances is None:
            return []

        return [
            RetrievedChunk(
                text=text,
                page_number=int(metadata["page_number"]),
                chunk_index=int(metadata["chunk_index"]),
                document_id=str(metadata["document_id"]),
                distance=float(distance),
            )
            for text, metadata, distance in zip(
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]
