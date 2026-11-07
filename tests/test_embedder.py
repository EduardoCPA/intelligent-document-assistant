import numpy as np
import pytest

from document_assistant.embeddings.embedder import FastEmbedder


class FakeModel:
    def __init__(self):
        self.received_texts = []
        self.call_count = 0

    def embed(self, texts):
        self.received_texts.append(texts)
        self.call_count += 1
        return (np.array([0.1, 0.2, 0.3], dtype=np.float32) for _ in texts)


def test_embed_documents_does_not_add_prefix():
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    embedder.embed_documents(["My document"])
    assert model.received_texts[0] == ["My document"]


def test_embed_query_does_not_add_prefix():
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    embedder.embed_query("What is AI?")
    assert model.received_texts[0] == ["What is AI?"]


def test_embeddings_are_converted_to_python_lists():
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    result = embedder.embed_documents(["Document one"])
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert all(isinstance(value, float) for value in result[0])
    assert result[0] == pytest.approx([0.1, 0.2, 0.3])


def test_empty_document_sequence_returns_empty_list():
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    assert embedder.embed_documents([]) == []
    assert model.call_count == 0


def test_blank_document_is_rejected():
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    with pytest.raises(ValueError, match="cannot be blank"):
        embedder.embed_documents(["Valid document", "   "])


def test_blank_query_is_rejected():
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    with pytest.raises(ValueError, match="Query cannot be blank"):
        embedder.embed_query("   ")


def test_model_is_reused():
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    embedder.embed_documents(["Document one"])
    embedder.embed_query("A question")
    assert model.call_count == 2


def test_number_of_embeddings_matches_number_of_documents():
    model = FakeModel()
    embedder = FastEmbedder(model=model)
    result = embedder.embed_documents(
        ["Document one", "Document two", "Document three"]
    )
    assert len(result) == 3


def test_mismatched_embedding_count_raises_error():
    class IncompleteModel:
        def embed(self, texts):
            return iter([])

    embedder = FastEmbedder(model=IncompleteModel())
    with pytest.raises(RuntimeError, match="does not match"):
        embedder.embed_documents(["Document one"])
