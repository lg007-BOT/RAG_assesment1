"""PDF extraction and sentence-aware chunking."""
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

from pypdf import PdfReader

from .config import CHUNK_OVERLAP_WORDS, CHUNK_TARGET_WORDS


@dataclass
class PageText:
    document: str
    page: int
    text: str


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pdf_pages(pdf_path: Path) -> Iterable[PageText]:
    """Yield non-empty pages with one-indexed page numbers."""
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalise_text(page.extract_text() or "")
        if text:
            yield PageText(document=pdf_path.name, page=page_number, text=text)


def split_sentences(text: str) -> List[str]:
    """A lightweight splitter adequate for prose PDFs without another NLP model."""
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalise_text(text))
    return [piece.strip() for piece in pieces if piece.strip()]


def is_heading(sentence: str) -> bool:
    """Conservative heading heuristic used only for useful trace metadata."""
    words = sentence.split()
    if not 1 <= len(words) <= 12 or sentence.endswith((".", "?", "!")):
        return False
    letters = [char for char in sentence if char.isalpha()]
    return bool(letters) and sum(char.isupper() for char in letters) / len(letters) > 0.7


def tail_by_words(sentences: List[str], word_limit: int) -> List[str]:
    selected: List[str] = []
    count = 0
    for sentence in reversed(sentences):
        selected.insert(0, sentence)
        count += len(sentence.split())
        if count >= word_limit:
            break
    return selected


def chunks_from_pages(pages: Iterable[PageText]) -> List[Dict[str, object]]:
    """Make chunks without crossing documents; page is the chunk's first page."""
    chunks: List[Dict[str, object]] = []
    current: List[str] = []
    current_words = 0
    current_document: Optional[str] = None
    current_page: Optional[int] = None
    latest_section: Optional[str] = None

    def emit() -> None:
        nonlocal current, current_words, current_page
        if not current or current_document is None or current_page is None:
            return
        chunk_number = len(chunks) + 1
        chunks.append({
            "id": "{}-p{}-c{}".format(Path(current_document).stem, current_page, chunk_number),
            "document": current_document,
            "page": current_page,
            "section": latest_section or "Unspecified",
            "text": " ".join(current),
        })
        current = tail_by_words(current, CHUNK_OVERLAP_WORDS)
        current_words = sum(len(sentence.split()) for sentence in current)
        current_page = None

    for page in pages:
        if current_document is not None and page.document != current_document:
            emit()
            current = []
            current_words = 0
        current_document = page.document
        for sentence in split_sentences(page.text):
            if is_heading(sentence):
                latest_section = sentence.title()
            word_count = len(sentence.split())
            if current and current_words + word_count > CHUNK_TARGET_WORDS:
                emit()
            if current_page is None:
                current_page = page.page
            current.append(sentence)
            current_words += word_count
    emit()
    return chunks
