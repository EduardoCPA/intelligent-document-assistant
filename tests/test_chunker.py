"""Tests for splitting extracted PDF pages into text chunks."""

import pytest

from document_assistant.ingestion.chunker import TextChunk, chunk_pages
from document_assistant.ingestion.pdf_loader import PdfPageText


def test_text_shorter_than_chunk_size_yields_one_chunk() -> None:
    pages = [PdfPageText(page_number=1, text="Short page text.")]

    chunks = chunk_pages(pages)

    assert chunks == [
        TextChunk(text="Short page text.", page_number=1, chunk_index=1),
    ]


def test_long_text_yields_multiple_chunks() -> None:
    text = " ".join(f"word{index:02d}" for index in range(40))

    chunks = chunk_pages(
        [PdfPageText(page_number=1, text=text)],
        chunk_size=40,
        overlap=10,
    )

    assert len(chunks) > 1
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_overlap_is_applied_between_chunks_of_the_same_page() -> None:
    text = "alpha bravo charlie delta echo foxtrot"

    chunks = chunk_pages(
        [PdfPageText(page_number=1, text=text)],
        chunk_size=20,
        overlap=10,
    )

    assert [chunk.text for chunk in chunks] == [
        "alpha bravo charlie",
        "charlie delta echo",
        "delta echo foxtrot",
    ]
    assert chunks[1].text.startswith("charlie")
    assert chunks[0].text.endswith("charlie")
    assert chunks[2].text.startswith("delta echo")
    assert chunks[1].text.endswith("delta echo")


def test_zero_overlap_does_not_repeat_words() -> None:
    text = "alpha bravo charlie delta"

    chunks = chunk_pages(
        [PdfPageText(page_number=1, text=text)],
        chunk_size=11,
        overlap=0,
    )

    assert [chunk.text for chunk in chunks] == ["alpha bravo", "charlie", "delta"]


def test_words_are_not_split() -> None:
    text = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod"
    words = text.split()

    chunks = chunk_pages(
        [PdfPageText(page_number=4, text=text)],
        chunk_size=28,
        overlap=8,
    )

    assert len(chunks) > 1
    for chunk in chunks:
        chunk_words = chunk.text.split()
        assert _is_contiguous_slice(chunk_words, words)
        assert len(chunk.text) <= 28


def test_original_page_numbers_are_preserved() -> None:
    pages = [
        PdfPageText(page_number=2, text="alpha bravo charlie"),
        PdfPageText(page_number=7, text="delta echo foxtrot golf"),
    ]

    chunks = chunk_pages(pages, chunk_size=12, overlap=0)

    assert [(chunk.page_number, chunk.text) for chunk in chunks] == [
        (2, "alpha bravo"),
        (2, "charlie"),
        (7, "delta echo"),
        (7, "foxtrot golf"),
    ]


def test_chunk_indexes_are_sequential_across_the_document() -> None:
    pages = [
        PdfPageText(page_number=1, text="alpha bravo charlie"),
        PdfPageText(page_number=2, text="   "),
        PdfPageText(page_number=3, text="delta echo foxtrot golf"),
    ]

    chunks = chunk_pages(pages, chunk_size=12, overlap=0)

    assert [chunk.chunk_index for chunk in chunks] == [1, 2, 3, 4]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 3, 3]


def test_blank_pages_are_ignored() -> None:
    pages = [
        PdfPageText(page_number=1, text=""),
        PdfPageText(page_number=2, text="   \n\t  "),
        PdfPageText(page_number=3, text="kept"),
        PdfPageText(page_number=4, text=" \n "),
    ]

    chunks = chunk_pages(pages)

    assert chunks == [TextChunk(text="kept", page_number=3, chunk_index=1)]


def test_overlap_does_not_cross_pages() -> None:
    pages = [
        PdfPageText(page_number=1, text="alpha bravo charlie delta echo"),
        PdfPageText(page_number=2, text="foxtrot golf hotel india"),
    ]

    chunks = chunk_pages(pages, chunk_size=20, overlap=10)

    page_two = [chunk for chunk in chunks if chunk.page_number == 2]
    assert page_two[0].text == "foxtrot golf hotel"
    assert "echo" not in page_two[0].text.split()


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "chunk_size must be greater than zero"),
        (-5, 0, "chunk_size must be greater than zero"),
        (100, -1, "overlap must be zero or greater"),
        (100, 100, "overlap must be smaller than chunk_size"),
        (100, 150, "overlap must be smaller than chunk_size"),
    ],
)
def test_invalid_parameters_raise_value_error(
    chunk_size: int,
    overlap: int,
    message: str,
) -> None:
    pages = [PdfPageText(page_number=1, text="hello world")]

    with pytest.raises(ValueError, match=message):
        chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)


def test_no_chunk_exceeds_chunk_size() -> None:
    words = ["short", "mediumword", "tiny", "another", "token", "payload"] * 8
    text = " ".join(words)
    chunk_size = 35

    chunks = chunk_pages(
        [PdfPageText(page_number=1, text=text)],
        chunk_size=chunk_size,
        overlap=12,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= chunk_size for chunk in chunks)
    assert all(chunk.text.split() for chunk in chunks)


def test_whitespace_runs_are_collapsed_between_words() -> None:
    page = PdfPageText(page_number=1, text="  alpha   bravo\ncharlie\t")

    chunks = chunk_pages([page], chunk_size=100, overlap=0)

    assert chunks == [
        TextChunk(text="alpha bravo charlie", page_number=1, chunk_index=1),
    ]


def test_oversized_token_is_sliced_without_exceeding_chunk_size() -> None:
    page = PdfPageText(page_number=1, text="ok SUPERCALIFRAGILISTIC end")

    chunks = chunk_pages([page], chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "ok",
        "SUPERC",
        "RCALIF",
        "IFRAGI",
        "GILIST",
        "STIC",
        "end",
    ]
    assert all(len(chunk.text) <= 6 for chunk in chunks)


def _is_contiguous_slice(needle: list[str], haystack: list[str]) -> bool:
    window = len(needle)
    if window == 0 or window > len(haystack):
        return False
    return any(
        haystack[index : index + window] == needle
        for index in range(len(haystack) - window + 1)
    )
