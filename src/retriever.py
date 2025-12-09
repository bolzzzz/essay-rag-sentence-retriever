import asyncio
from typing import List, Tuple, Optional
from pydantic import BaseModel, ValidationError
from .pdf_utils import extract_text_with_metadata, get_page_texts
from .embedding import EmbeddingClient
from .vector_store import LocalVectorStore
from .key_claims import extract_key_claims_local
from .key_claims import split_into_paragraphs
from pydantic import BaseModel, Field
import re


class RetrieveInput(BaseModel):
    student_essay: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class RetrievalResult(BaseModel):
    sentence: str
    score: float = Field(ge=0.0, le=1.0)
    context: Optional[str] = None
    chapter: Optional[str] = None
    page: Optional[int] = None


class SentenceRetriever:
    def __init__(self, book_path: str, store: Optional[LocalVectorStore] = None):
        self.book_path = book_path
        self._store = store or LocalVectorStore()
        self._initialized = False
        self._corpus: List[Tuple[str, Optional[str], Optional[int]]] = []
        self._collection = "sentences"

    async def ainit(self):
        # Build sentence_id -> sentence map consistent with indexer paragraph segmentation
        page_texts = await asyncio.to_thread(get_page_texts, self.book_path)
        def split_into_paragraphs(text: str) -> List[str]:
            import re as _re
            parts = _re.split(r"\n\s*\n+", (text or "").strip())
            return [p.strip() for p in parts if p.strip()]
        # Use the same sentence splitter
        from .pdf_utils import split_into_sentences as _split_sents
        self._id_to_sentence: dict[int, str] = {}
        sentence_counter = 0
        for page_text in page_texts:
            paragraphs = split_into_paragraphs(page_text)
            for para in paragraphs:
                para_sents = _split_sents(para)
                for s in para_sents:
                    s_clean = s.strip()
                    if s_clean:
                        self._id_to_sentence[sentence_counter] = s_clean
                    sentence_counter += 1
        self._initialized = True

    async def retrieve(self, student_essay: str, top_k: int = 5) -> List[RetrievalResult]:
        try:
            req = RetrieveInput(student_essay=student_essay, top_k=top_k)
        except ValidationError as e:
            raise e

        if not self._initialized:
            await self.ainit()

        # Multi-level queries: paragraph-level and key-claim-level
        paragraphs = split_into_paragraphs(req.student_essay)
        key_claims = await extract_key_claims_local(req.student_essay, max_claims=max(1, len(paragraphs)))
        async with EmbeddingClient() as ec:
            q_emb = await ec.embed(paragraphs + key_claims)
        # Query per vector and collect candidates (use Chroma distances in cosine space)
        candidates: List[Tuple[str, dict, float]] = []
        per_query_n = 2
        for i in range(len(q_emb)):
            res = self._store.query(collection=self._collection, query_embeddings=[q_emb[i]], n_results=per_query_n)
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                candidates.append((doc, meta, float(dist)))
        print(f"Total candidates: {len(candidates)}")

        # Diversity filtering and score fusion
        results: List[RetrievalResult] = []
        # Score fusion using Chroma distances (cosine space): similarity = 1 - distance
        # If a sentence appears multiple times, keep the largest similarity score.
        best: dict[str, Tuple[dict, float]] = {}
        for doc, meta, dist in candidates:
            score = 1.0 - dist
            prev = best.get(doc)
            if prev is None or score > prev[1]:
                best[doc] = (meta, score)
        fused: List[Tuple[str, dict, float]] = [(doc, meta, sc) for doc, (meta, sc) in best.items()]

        fused.sort(key=lambda x: x[2], reverse=True)
        # Diversity filtering by approximate semantic overlap using tokens
        diverse: List[Tuple[str, dict, float]] = []
        seen_token_sets: List[Set[str]] = []
        for doc, meta, score in fused:
            ts = _tokenize(doc)
            too_similar = any(len(ts & s) / max(1, len(ts | s)) > 0.9 for s in seen_token_sets)
            if too_similar:
                continue
            seen_token_sets.append(ts)
            diverse.append((doc, meta, score))
        print(f"After score fusion and diversity filtering: {len(diverse)}")

        selected = diverse[:req.top_k]
        from .pdf_utils import sanitize_text as _sanitize
        for doc, meta, score in selected:
            chapter = meta.get("chapter") if isinstance(meta, dict) else None
            page = meta.get("page") if isinstance(meta, dict) else None
            # Build context using prev/next sentence IDs from metadata
            context = None
            prev_id = meta.get("prev_sentence_id") if isinstance(meta, dict) else None
            next_id = meta.get("next_sentence_id") if isinstance(meta, dict) else None
            prev_s = None if prev_id in (None, "null") else self._id_to_sentence.get(prev_id)
            next_s = None if next_id in (None, "null") else self._id_to_sentence.get(next_id)
            prev_part = prev_s if prev_s else ""
            next_part = next_s if next_s else ""
            context = f"{prev_part} --{doc}-- {next_part}"
            results.append(RetrievalResult(sentence=_sanitize(doc), score=round(score, 4), context=context, chapter=chapter, page=page))

        return results


from typing import Set


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return set(tokens)
