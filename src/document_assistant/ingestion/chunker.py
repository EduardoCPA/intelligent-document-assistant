"""Split extracted PDF pages into overlapping, word-aligned chunks."""

from collections.abc import Sequence
from dataclasses import dataclass

from document_assistant.ingestion.pdf_loader import PdfPageText


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A slice of page text small enough for later embedding.

    Attributes:
        text: Chunk content. Tokens are separated by a single space.
        page_number: Page the text came from, copied from ``PdfPageText``.
        chunk_index: Position of this chunk in the document, starting at 1.
    """

    text: str
    page_number: int
    chunk_index: int


def chunk_pages(
    pages: Sequence[PdfPageText],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split each page into chunks of at most ``chunk_size`` characters.

    Each page is chunked independently. Normal chunks share a word-aligned
    suffix of up to ``overlap`` characters with the next chunk. If a token
    is longer than ``chunk_size``, it is split into character-based pieces
    that overlap each other. No overlap is added between a normal chunk
    and an oversized token.

    Whitespace-only pages are ignored. On the pages that remain, leading and
    trailing whitespace is discarded and any run of whitespace becomes a single
    space, so chunk boundaries fall between words.

    A token longer than ``chunk_size`` cannot sit on a word boundary inside the
    limit. That token is the only case split by characters. Those pieces still
    respect ``chunk_size`` and repeat ``overlap`` characters between them.

    Args:
        pages: Extracted pages, in the order they should be chunked.
        chunk_size: Maximum chunk length, in characters. Must be greater than
            zero.
        overlap: Maximum number of characters repeated between consecutive
            chunks of one page. Must be zero or greater, and smaller than
            ``chunk_size``.

    Returns:
        Chunks in reading order. Empty when every page is blank.

    Raises:
        ValueError: ``chunk_size`` or ``overlap`` is outside the allowed range.
    """
    _validate_parameters(chunk_size, overlap)

    chunks: list[TextChunk] = []
    for page in pages:
        for piece in _chunk_text(page.text, chunk_size, overlap):
            chunks.append(
                TextChunk(
                    text=piece,
                    page_number=page.page_number,
                    chunk_index=len(chunks) + 1,
                )
            )
    return chunks


def _validate_parameters(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than zero, got {chunk_size}.")
    if overlap < 0:
        raise ValueError(f"overlap must be zero or greater, got {overlap}.")
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size, "
            f"got overlap={overlap} and chunk_size={chunk_size}."
        )


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    index = 0

    while index < len(words):
        word = words[index]
        if len(word) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.extend(_split_oversized_word(word, chunk_size, overlap))
            index += 1
            continue

        word_would_overflow = (
            bool(current) and _joined_length(current) + 1 + len(word) > chunk_size
        )
        if word_would_overflow:
            chunks.append(" ".join(current))
            current = _fitting_overlap(current, word, chunk_size, overlap)
            continue

        current.append(word)
        index += 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def _joined_length(tokens: Sequence[str]) -> int:
    """Return the length of tokens joined by single spaces."""
    if not tokens:
        return 0
    return sum(len(token) for token in tokens) + len(tokens) - 1


def _fitting_overlap(
    tokens: Sequence[str],
    next_word: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Return a word-aligned suffix that still leaves room for ``next_word``.

    The suffix is the longest run of trailing tokens whose joined length is at
    most ``overlap``. Tokens are then dropped from the front while the suffix
    plus ``next_word`` would exceed ``chunk_size``. The caller retries
    ``next_word``, so the next chunk always gains at least that new token.
    """
    selected: list[str] = []
    if overlap > 0:
        for token in reversed(tokens):
            candidate = [token, *selected]
            if _joined_length(candidate) > overlap:
                break
            selected = candidate

    while selected and _joined_length(selected) + 1 + len(next_word) > chunk_size:
        selected.pop(0)
    return selected


def _split_oversized_word(word: str, chunk_size: int, overlap: int) -> list[str]:
    """Slice a token that cannot fit in one chunk.

    ``overlap`` is smaller than ``chunk_size``, so the window advances by at
    least one character and the loop terminates.
    """
    pieces: list[str] = []
    step = chunk_size - overlap
    start = 0
    limit = len(word)
    while start < limit:
        pieces.append(word[start : start + chunk_size])
        if start + chunk_size >= limit:
            break
        start += step
    return pieces
