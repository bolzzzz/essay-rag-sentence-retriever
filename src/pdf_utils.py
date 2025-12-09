import re
from typing import List, Optional, Tuple
from pypdf import PdfReader


def extract_text_with_metadata(pdf_path: str) -> List[Tuple[str, Optional[str], Optional[int]]]:
    reader = PdfReader(pdf_path)
    results: List[Tuple[str, Optional[str], Optional[int]]] = []
    chapter: Optional[str] = None

    chapter_pattern = re.compile(r"^(Chapter|CHAPTER)\s+([\w\d\s:-]+)$")

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        # naive chapter detection: look for lines starting with Chapter
        for line in text.splitlines():
            m = chapter_pattern.match(line.strip())
            if m:
                chapter = m.group(0)
                break

        sentences = split_into_sentences(text)
        for s in sentences:
            s_clean = s.strip()
            if s_clean:
                results.append((s_clean, chapter, page_num))

    return results


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    # simple sentence splitter handling ., ?, ! and quotes
    parts = re.split(r"(?<=[\.?\!])\s+\"?", text)
    return [p for p in parts if p]


def get_page_texts(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        texts.append(t)
    return texts


# TXT input support has been removed; this repository targets PDF-only.
