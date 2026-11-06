from document_assistant.embeddings.embedder import FastEmbedder
from document_assistant.retrieval.vector_store import (
    ChromaVectorStore,
    RetrievedChunk,
)


class Retriever:
    """Retrieve relevant document chunks for a natural language query."""

    def __init__(
        self,
        embedder: FastEmbedder | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        """Initialize the retriever with its dependencies."""
        self._embedder = embedder if embedder is not None else FastEmbedder()
        self._vector_store = (
            vector_store if vector_store is not None else ChromaVectorStore()
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a natural language query."""
        query_embedding = self._embedder.embed_query(query)

        return self._vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            document_id=document_id,
        )
