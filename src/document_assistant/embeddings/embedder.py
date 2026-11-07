from collections.abc import Sequence
from typing import Any

from fastembed import TextEmbedding

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedder:
    """Generate document and query embeddings using FastEmbed."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Any | None = None,
    ) -> None:
        """Initialize the embedding model once per instance."""
        self._model = (
            model if model is not None else TextEmbedding(model_name=model_name)
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for document texts."""
        if not texts:
            return []

        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"Document text at index {index} cannot be blank.")

        prefixed_texts = [text.strip() for text in texts]
        vectors = list(self._model.embed(prefixed_texts))

        if len(vectors) != len(texts):
            raise RuntimeError(
                "The number of generated embeddings does not match "
                "the number of input documents."
            )

        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a search query."""
        if not query.strip():
            raise ValueError("Query cannot be blank.")

        vectors = list(self._model.embed([query.strip()]))

        if len(vectors) != 1:
            raise RuntimeError("Expected exactly one embedding for the query.")

        return vectors[0].tolist()
