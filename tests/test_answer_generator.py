from unittest.mock import Mock

import pytest

from document_assistant.generation.answer_generator import (
    ABSTENTION_MESSAGE,
    AnswerGenerator,
    GeneratedAnswer,
    SourceCitation,
)
from document_assistant.retrieval.vector_store import RetrievedChunk


@pytest.fixture
def mock_retriever():
    retriever = Mock()
    retriever.retrieve.return_value = [
        RetrievedChunk(
            text="Python é uma linguagem de programação.",
            page_number=2,
            chunk_index=0,
            document_id="document-1",
            distance=0.15,
        ),
        RetrievedChunk(
            text="Python possui tipagem dinâmica.",
            page_number=3,
            chunk_index=1,
            document_id="document-1",
            distance=0.20,
        ),
    ]
    return retriever


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.generate.return_value = "Python é uma linguagem de programação. [S1]"
    return llm


@pytest.fixture
def answer_generator(mock_retriever, mock_llm):
    return AnswerGenerator(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )


def test_answer_question_returns_generated_answer(answer_generator):
    result = answer_generator.answer_question("O que é Python?")

    assert isinstance(result, GeneratedAnswer)
    assert result.answer == ("Python é uma linguagem de programação. [S1]")


def test_answer_question_returns_only_cited_sources(answer_generator):
    result = answer_generator.answer_question("O que é Python?")

    assert result.sources == (SourceCitation("document-1", 2, 0),)


def test_answer_question_passes_question_to_retriever(
    answer_generator,
    mock_retriever,
):
    answer_generator.answer_question("O que é Python?")

    mock_retriever.retrieve.assert_called_once_with(
        query="O que é Python?",
        n_results=5,
        document_id=None,
    )


def test_answer_question_passes_custom_retrieval_parameters(
    answer_generator,
    mock_retriever,
):
    answer_generator.answer_question(
        "O que é Python?",
        n_results=3,
        document_id="document-1",
    )

    mock_retriever.retrieve.assert_called_once_with(
        query="O que é Python?",
        n_results=3,
        document_id="document-1",
    )


def test_answer_question_includes_context_in_prompt(
    answer_generator,
    mock_llm,
):
    answer_generator.answer_question("O que é Python?")

    prompt = mock_llm.generate.call_args.args[0]

    assert "Python é uma linguagem de programação." in prompt
    assert "Python possui tipagem dinâmica." in prompt
    assert "[S1 | Page 2]" in prompt
    assert "[S2 | Page 3]" in prompt


def test_answer_question_includes_grounding_instructions(
    answer_generator,
    mock_llm,
):
    answer_generator.answer_question("O que é Python?")

    prompt = mock_llm.generate.call_args.args[0]

    assert "ONLY the provided document context" in prompt
    assert "untrusted data" in prompt
    assert "Do not invent source identifiers" in prompt


def test_answer_question_abstains_when_no_chunks_are_found(
    mock_retriever,
    mock_llm,
):
    mock_retriever.retrieve.return_value = []

    generator = AnswerGenerator(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    result = generator.answer_question("Quem inventou Python?")

    assert result.answer == ABSTENTION_MESSAGE
    assert result.sources == ()
    mock_llm.generate.assert_not_called()


def test_answer_question_rejects_blank_question(
    answer_generator,
    mock_retriever,
):
    with pytest.raises(ValueError, match="Question cannot be blank"):
        answer_generator.answer_question("   ")

    mock_retriever.retrieve.assert_not_called()


def test_answer_question_propagates_retrieval_errors(
    mock_retriever,
    mock_llm,
):
    mock_retriever.retrieve.side_effect = RuntimeError("Retrieval failed")

    generator = AnswerGenerator(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    with pytest.raises(RuntimeError, match="Retrieval failed"):
        generator.answer_question("Question")

    mock_llm.generate.assert_not_called()


def test_answer_question_propagates_generation_errors(
    mock_retriever,
    mock_llm,
):
    mock_llm.generate.side_effect = RuntimeError("Generation failed")

    generator = AnswerGenerator(
        retriever=mock_retriever,
        llm_client=mock_llm,
    )

    with pytest.raises(RuntimeError, match="Generation failed"):
        generator.answer_question("Question")

    mock_retriever.retrieve.assert_called_once()


def test_answer_question_returns_multiple_cited_sources(
    answer_generator,
    mock_llm,
):
    mock_llm.generate.return_value = "Python é uma linguagem. [S1] [S2]"

    result = answer_generator.answer_question("O que é Python?")

    assert result.sources == (
        SourceCitation("document-1", 2, 0),
        SourceCitation("document-1", 3, 1),
    )


def test_answer_question_ignores_invalid_citations(
    answer_generator,
    mock_llm,
):
    mock_llm.generate.return_value = "Python é uma linguagem. [S1] [S99]"

    result = answer_generator.answer_question("O que é Python?")

    assert result.sources == (SourceCitation("document-1", 2, 0),)
    assert "[S99]" not in result.answer


def test_answer_question_removes_duplicate_source_citations(
    answer_generator,
    mock_llm,
):
    mock_llm.generate.return_value = "Python é uma linguagem. [S1] [S1]"

    result = answer_generator.answer_question("O que é Python?")

    assert result.sources == (SourceCitation("document-1", 2, 0),)


def test_answer_question_abstains_without_valid_citations(
    answer_generator,
    mock_llm,
):
    mock_llm.generate.return_value = "Python é uma linguagem, mas sem referências."

    result = answer_generator.answer_question("O que é Python?")

    assert result.answer == ABSTENTION_MESSAGE
    assert result.sources == ()
