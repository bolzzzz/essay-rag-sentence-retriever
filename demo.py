import asyncio
import os
import json
from src.retriever import SentenceRetriever
from src.indexer import OfflineIndexer
from src.vector_store import LocalVectorStore
from src.embedding import EmbeddingClient


async def main():
    job_path = os.path.join("jobs", "demo")
    os.makedirs(job_path, exist_ok=True)
    book_path = os.path.join(job_path, "2025_12_06_Digital Minimalism by Cal Newport .pdf")
    essay_path = os.path.join(job_path, "essay_sample.txt")
    
    top_k = 5

    store = LocalVectorStore()
    indexer = OfflineIndexer(store)
    await indexer.build_index(book_path)
    retriever = SentenceRetriever(book_path=book_path)
    with open(essay_path, "r", encoding="utf-8") as f:
        essay = f.read()
    results = await retriever.retrieve(student_essay=essay, top_k=top_k)
    print(f"top {top_k} sentences:")
    print('===')
    for r in results:
        print(f"Sentence: {r.sentence}")
        print(f"Score: {r.score}")
        print(f"Context: {r.context}")
        print(f"Location: Chapter {r.chapter}, Page {r.page}")
        print("---")
    out_path = os.path.join(job_path, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
