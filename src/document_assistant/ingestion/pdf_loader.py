"""Extract text from PDF files, one page at a time."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PyPdfError, WrongPasswordError


class PdfLoaderError(Exception):
    """Base error raised while reading a PDF."""


class InvalidPdfError(PdfLoaderError):
    """The file exists but cannot be read as a PDF."""


class EncryptedPdfError(PdfLoaderError):
    """The PDF is encrypted and no password was provided."""


class EmptyPdfError(PdfLoaderError):
    """The PDF does not contain any pages."""


@dataclass(frozen=True, slots=True)
class PdfPageText:
    """Text extracted from a single PDF page.

    Attributes:
        page_number: Page position in the document, starting at 1.
        text: Extracted text. Empty when the page has no text content.
    """

    page_number: int
    text: str


def extract_text_from_pdf(pdf_path: str | Path) -> list[PdfPageText]:
    """Extract the text of each page from a PDF file.

    Args:
        pdf_path: Location of the PDF. Accepts a string or a ``Path``.

    Returns:
        One ``PdfPageText`` per page, in document order. Page numbers start
        at 1. A page with no extractable text is returned with ``text=""``.

    Raises:
        FileNotFoundError: ``pdf_path`` does not point to an existing file.
        ValueError: The file extension is not ``.pdf``.
        EncryptedPdfError: The PDF is encrypted, or a page cannot be decrypted.
        EmptyPdfError: The PDF has no pages.
        InvalidPdfError: The file is not a readable PDF, or a page cannot be read.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got '{path.suffix or path.name}'.")

    try:
        reader = PdfReader(path)
        encrypted = reader.is_encrypted
    except (FileNotDecryptedError, WrongPasswordError) as error:
        raise EncryptedPdfError(
            f"PDF is encrypted and cannot be read without a password: {path.name}"
        ) from error
    except PyPdfError as error:
        raise InvalidPdfError(f"Invalid PDF file: {path.name}") from error

    if encrypted:
        raise EncryptedPdfError(
            f"PDF is encrypted and cannot be read without a password: {path.name}"
        )

    try:
        page_count = len(reader.pages)
    except (FileNotDecryptedError, WrongPasswordError) as error:
        raise EncryptedPdfError(
            f"PDF is encrypted and cannot be read without a password: {path.name}"
        ) from error
    except PyPdfError as error:
        raise InvalidPdfError(f"Invalid PDF file: {path.name}") from error

    if page_count == 0:
        raise EmptyPdfError(f"PDF has no pages: {path.name}")

    extracted_pages: list[PdfPageText] = []
    for page_number in range(1, page_count + 1):
        try:
            raw_text = reader.pages[page_number - 1].extract_text()
        except (FileNotDecryptedError, WrongPasswordError) as error:
            raise EncryptedPdfError(
                f"Could not read encrypted page {page_number} of '{path.name}'."
            ) from error
        except PyPdfError as error:
            raise InvalidPdfError(
                f"Could not read page {page_number} of '{path.name}'."
            ) from error

        extracted_pages.append(
            PdfPageText(page_number=page_number, text=raw_text or "")
        )

    return extracted_pages
