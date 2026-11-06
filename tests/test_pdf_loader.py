"""Tests for PDF text extraction."""

from pathlib import Path

import pytest
from pypdf.errors import FileNotDecryptedError, PdfReadError

from document_assistant.ingestion.pdf_loader import (
    EmptyPdfError,
    EncryptedPdfError,
    InvalidPdfError,
    PdfPageText,
    extract_text_from_pdf,
)


class _FakePage:
    def __init__(
        self,
        text: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self._text = text
        self._error = error

    def extract_text(self) -> str | None:
        if self._error is not None:
            raise self._error
        return self._text


class _FakeReader:
    def __init__(
        self,
        path: str | Path,
        pages: list[_FakePage],
        *,
        is_encrypted: bool = False,
    ) -> None:
        self.path = path
        self._pages = pages
        self.is_encrypted = is_encrypted
        self.pages_accessed = False

    @property
    def pages(self) -> list[_FakePage]:
        self.pages_accessed = True
        return self._pages


def _touch_pdf(directory: Path, name: str = "document.pdf") -> Path:
    pdf_path = directory / name
    pdf_path.write_bytes(b"%PDF-1.4")
    return pdf_path


def test_extracts_text_from_multiple_pages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = _touch_pdf(tmp_path)
    pages = [
        _FakePage("First page"),
        _FakePage(None),
        _FakePage("Third page"),
    ]
    monkeypatch.setattr(
        "document_assistant.ingestion.pdf_loader.PdfReader",
        lambda path: _FakeReader(path, pages),
    )

    extracted = extract_text_from_pdf(pdf_path)

    assert extracted == [
        PdfPageText(page_number=1, text="First page"),
        PdfPageText(page_number=2, text=""),
        PdfPageText(page_number=3, text="Third page"),
    ]


def test_preserves_original_page_numbers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = _touch_pdf(tmp_path)
    pages = [_FakePage("A"), _FakePage("B"), _FakePage(""), _FakePage("D")]
    monkeypatch.setattr(
        "document_assistant.ingestion.pdf_loader.PdfReader",
        lambda path: _FakeReader(path, pages),
    )

    extracted = extract_text_from_pdf(str(pdf_path))

    assert [page.page_number for page in extracted] == [1, 2, 3, 4]
    assert extracted[2].text == ""


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        extract_text_from_pdf(missing)


def test_invalid_extension_raises_value_error(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a .pdf file"):
        extract_text_from_pdf(notes)


def test_invalid_pdf_raises_invalid_pdf_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = _touch_pdf(tmp_path, "broken.pdf")

    def _raise_invalid(path: str | Path) -> _FakeReader:
        raise PdfReadError("EOF marker not found")

    monkeypatch.setattr(
        "document_assistant.ingestion.pdf_loader.PdfReader",
        _raise_invalid,
    )

    with pytest.raises(InvalidPdfError, match="Invalid PDF file"):
        extract_text_from_pdf(pdf_path)


def test_encrypted_pdf_raises_encrypted_pdf_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = _touch_pdf(tmp_path, "secret.pdf")

    def _raise_encrypted(path: str | Path) -> _FakeReader:
        raise FileNotDecryptedError("File has not been decrypted")

    monkeypatch.setattr(
        "document_assistant.ingestion.pdf_loader.PdfReader",
        _raise_encrypted,
    )

    with pytest.raises(EncryptedPdfError, match="encrypted"):
        extract_text_from_pdf(pdf_path)


def test_encrypted_pdf_is_rejected_before_page_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = _touch_pdf(tmp_path, "locked.pdf")
    reader = _FakeReader(
        pdf_path,
        [_FakePage("secret")],
        is_encrypted=True,
    )
    monkeypatch.setattr(
        "document_assistant.ingestion.pdf_loader.PdfReader",
        lambda path: reader,
    )

    with pytest.raises(EncryptedPdfError, match="encrypted"):
        extract_text_from_pdf(pdf_path)

    assert reader.pages_accessed is False


def test_extract_text_file_not_decrypted_reports_page_number(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = _touch_pdf(tmp_path, "partial.pdf")
    pages = [
        _FakePage("visible"),
        _FakePage(error=FileNotDecryptedError("File has not been decrypted")),
    ]
    monkeypatch.setattr(
        "document_assistant.ingestion.pdf_loader.PdfReader",
        lambda path: _FakeReader(path, pages, is_encrypted=False),
    )

    with pytest.raises(EncryptedPdfError, match="page 2"):
        extract_text_from_pdf(pdf_path)


def test_pdf_without_pages_raises_empty_pdf_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_path = _touch_pdf(tmp_path, "empty.pdf")
    monkeypatch.setattr(
        "document_assistant.ingestion.pdf_loader.PdfReader",
        lambda path: _FakeReader(path, []),
    )

    with pytest.raises(EmptyPdfError, match="no pages"):
        extract_text_from_pdf(pdf_path)
