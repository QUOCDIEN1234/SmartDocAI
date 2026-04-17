from pathlib import Path

import docx
import pdfplumber


PDF_EXTRACT_X_TOLERANCE = 1.5
PDF_EXTRACT_Y_TOLERANCE = 3


def _ensure_file_exists(file_path: str, file_type: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} file not found: {file_path}")
    return path


def _extract_pdf_page_text(page) -> str:
    return (
        page.extract_text(
            x_tolerance=PDF_EXTRACT_X_TOLERANCE,
            y_tolerance=PDF_EXTRACT_Y_TOLERANCE,
        )
        or ""
    )


def load_pdf(file_path: str) -> str:
    """Read all text from a PDF file and return one combined string."""
    path = _ensure_file_exists(file_path, "PDF")

    pages_text = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = _extract_pdf_page_text(page).strip()
            pages_text.append(page_text)

    return "\n\n".join(p for p in pages_text if p)


def load_pdf_pages(file_path: str) -> list:
    """Read PDF page-by-page to support source attribution (page + context)."""
    path = _ensure_file_exists(file_path, "PDF")

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_text = _extract_pdf_page_text(page).strip()
            if page_text:
                pages.append({"page": page_idx, "text": page_text})

    return pages


def load_docx(file_path: str) -> str:
    """Read all paragraph text from a DOCX file."""
    path = _ensure_file_exists(file_path, "DOCX")

    doc = docx.Document(str(path))
    full_text = [para.text for para in doc.paragraphs]
    return "\n".join(full_text)
