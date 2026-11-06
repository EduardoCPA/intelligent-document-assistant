from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from document_assistant.generation.llm_client import GeminiClient


@pytest.fixture
def mock_client():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text="Generated answer"
    )
    return client


def test_generate_returns_model_response(mock_client):
    gemini = GeminiClient(
        client=mock_client,
        model_name="test-model",
    )

    result = gemini.generate("What is artificial intelligence?")

    assert result == "Generated answer"


def test_generate_calls_model_with_correct_parameters(mock_client):
    gemini = GeminiClient(
        client=mock_client,
        model_name="test-model",
    )

    gemini.generate("Explain machine learning.")

    mock_client.models.generate_content.assert_called_once_with(
        model="test-model",
        contents="Explain machine learning.",
    )


def test_generate_strips_response_whitespace(mock_client):
    mock_client.models.generate_content.return_value.text = "  Answer  "

    gemini = GeminiClient(client=mock_client)

    result = gemini.generate("Question")

    assert result == "Answer"


def test_generate_rejects_blank_prompt(mock_client):
    gemini = GeminiClient(client=mock_client)

    with pytest.raises(ValueError, match="Prompt cannot be blank"):
        gemini.generate("   ")

    mock_client.models.generate_content.assert_not_called()


def test_generate_rejects_empty_response(mock_client):
    mock_client.models.generate_content.return_value.text = None

    gemini = GeminiClient(client=mock_client)

    with pytest.raises(RuntimeError, match="empty response"):
        gemini.generate("Question")


def test_client_requires_api_key_when_no_client_is_injected(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient()
