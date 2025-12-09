from typing import List, Dict
import numpy as np
import chromadb
from chromadb.config import Settings


class LocalVectorStore:
    def __init__(self):
        self._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self._collections: Dict[str, any] = {}

    def ensure_collection(self, name: str):
        if name in self._collections:
            return self._collections[name]
        col = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self._collections[name] = col
        return col

    def add(self, collection: str, ids: List[str], embeddings: List[List[float]], metadatas: List[dict], documents: List[str]):
        col = self.ensure_collection(collection)
        col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, collection: str, query_embeddings: List[List[float]], n_results: int):
        col = self.ensure_collection(collection)
        res = col.query(query_embeddings=query_embeddings, n_results=n_results, include=["metadatas", "documents", "distances"]) 
        return res

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
