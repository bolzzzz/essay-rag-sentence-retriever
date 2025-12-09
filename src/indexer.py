import asyncio
import logging
from typing import List, Optional, Tuple
from .pdf_utils import extract_text_with_metadata
from .embedding import EmbeddingClient
from .vector_store import LocalVectorStore
from logging import StreamHandler, Formatter
from .pdf_utils import get_page_texts


def classify_page(text: str):
    t = (text or "").lower()
    rules = [
        ("references", r"^\s*references\b|\bdoi:\b|\[[0-9]+\]"),
        ("notes", r"^\s*notes\b|^\s*footnotes\b|\bnote:\b"),
        ("acknowledgements", r"^\s*acknowledgements\b|^\s*acknowledgments\b"),
        ("copyright", r"©|copyright|all rights reserved|isbn"),
        ("metadata", r"library of congress|publisher|edition|printing|cover design|typeset by"),
    ]
    import re
    for label, pat in rules:
        if re.search(pat, t):
            return (label, 0.9)
    if len(t.strip()) < 200 or re.search(r"^\s*contents\b", t):
        return ("metadata", 0.6)
    return ("content", 0.8)


def is_sentence_content(sentence: str) -> bool:
    s = (sentence or "").strip()
    if len(s) < 5:
        return False
    import re
    if re.match(r"^\d+\s*$", s):
        return False
    if re.search(r"\([0-9]{4}\)|\bdoi:\b|\bet al\.\b", s.lower()):
        return False
    return True


logger = logging.getLogger("indexer")
if not logger.handlers:
    handler = StreamHandler()
    handler.setFormatter(Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class OfflineIndexer:
    def __init__(self, store: LocalVectorStore | None = None, collection_name: str = "sentences"):
        self.store = store or LocalVectorStore()
        self.collection = collection_name

    async def build_index(self, pdf_path: str):
        try:
            # Build paragraph-aware unique IDs and prev/next per paragraph
            sentences_meta: List[Tuple[str, Optional[str], Optional[int]]] = await asyncio.to_thread(extract_text_with_metadata, pdf_path)
            page_texts = await asyncio.to_thread(get_page_texts, pdf_path)
            page_labels = [classify_page(t) for t in page_texts]

            # Organize sentences by page to align with paragraph segmentation
            by_page: dict[int, List[Tuple[str, Optional[str]]]] = {}
            for s, ch, pg in sentences_meta:
                p = int(pg or -1)
                by_page.setdefault(p, []).append((s, ch))

            def split_into_paragraphs(text: str) -> List[str]:
                import re as _re
                parts = _re.split(r"\n\s*\n+", (text or "").strip())
                return [p.strip() for p in parts if p.strip()]

            filtered_docs = []
            filtered_metas = []
            filtered_ids = []
            filtered_logs = []

            sentence_counter = 0
            paragraph_counter = 0

            for page_num, page_text in enumerate(page_texts, start=1):
                label, conf = page_labels[page_num - 1] if page_num - 1 < len(page_labels) else ("content", 0.5)
                # Paragraphs for this page
                paragraphs = split_into_paragraphs(page_text)
                # Pointer over by_page sentences for this page to preserve chapter assignment
                page_sent_list = by_page.get(page_num, [])
                ptr = 0
                for para in paragraphs:
                    # Split paragraph into sentences using same splitter
                    para_sents = extract_text_with_metadata.__globals__["split_into_sentences"](para)
                    # Assign unique paragraph id
                    paragraph_id = paragraph_counter
                    paragraph_counter += 1
                    for j, s in enumerate(para_sents):
                        from .pdf_utils import sanitize_text as _sanitize
                        s_clean = _sanitize(s)
                        # Find chapter from by_page list (advance pointer until match or end)
                        ch_for_s = "Unknown"
                        # advance ptr while mismatch
                        while ptr < len(page_sent_list) and page_sent_list[ptr][0].strip() != s_clean:
                            ptr += 1
                        if ptr < len(page_sent_list):
                            ch_for_s = page_sent_list[ptr][1] or "Unknown"
                            ptr += 1
                        # Compute prev/next per paragraph
                        prev_id = "null" if j == 0 else sentence_counter - 1
                        # next id is null if last in paragraph; note next is sentence_counter+1 (to be assigned next)
                        next_id = "null" if j == len(para_sents) - 1 else sentence_counter + 1

                        # Build metadata
                        meta = {
                            "book": pdf_path,
                            "chapter": ch_for_s,
                            "page": page_num,
                            "paragraph_id": paragraph_id,
                            "sentence_id": sentence_counter,
                            "prev_sentence_id": prev_id,
                            "next_sentence_id": next_id,
                        }

                        # Apply page-level and sentence-level filters
                        if label != "content" or not is_sentence_content(s_clean):
                            filtered_logs.append({"page": page_num, "label": label, "confidence": conf, "sentence": s_clean[:200]})
                        else:
                            filtered_docs.append(s_clean)
                            filtered_metas.append(meta)
                            filtered_ids.append(f"{pdf_path}-sent-{sentence_counter}")

                        sentence_counter += 1
            documents = filtered_docs
            metadatas = filtered_metas
            ids = filtered_ids

            async with EmbeddingClient() as ec:
                embeddings = await ec.embed(documents)

            self.store.add(collection=self.collection, ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        except Exception as e:
            logger.exception(f"Indexing failed for {pdf_path}: {e}")
            raise
