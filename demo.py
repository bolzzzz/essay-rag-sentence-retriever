import asyncio
from src.retriever import SentenceRetriever
from src.indexer import OfflineIndexer
from src.vector_store import LocalVectorStore
from src.embedding import EmbeddingClient


async def main():
    # Check embeddings server availability before indexing
    async with EmbeddingClient() as ec:
        await ec.embed(["healthcheck"])  # will raise if unreachable

    store = LocalVectorStore()
    indexer = OfflineIndexer(store)
    await indexer.build_index("2025_12_06_Digital Minimalism by Cal Newport .pdf")
    retriever = SentenceRetriever(book_path="2025_12_06_Digital Minimalism by Cal Newport .pdf")
    essay = "This is a sample essay text used to query the book."
    results = await retriever.retrieve(student_essay=essay, top_k=5)
    for r in results:
        print(f"Sentence: {r.sentence}")
        print(f"Score: {r.score}")
        print(f"Context: {r.context}")
        print(f"Location: Chapter {r.chapter}, Page {r.page}")
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
