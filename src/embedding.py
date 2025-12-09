import asyncio
from typing import List, Optional
import httpx
from pydantic import BaseModel, Field


class VLLMEmbeddingRequest(BaseModel):
    input: List[str]
    model: str = Field(default="intfloat/e5-small")


class VLLMEmbeddingItem(BaseModel):
    object: str
    embedding: List[float]
    index: int


class VLLMEmbeddingResponse(BaseModel):
    object: str
    data: List[VLLMEmbeddingItem]
    model: Optional[str] = None
    usage: Optional[dict] = None


class EmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            await self._client.aclose()
        self._client = None

    async def embed(self, texts: List[str]) -> List[List[float]]:
        assert self._client is not None, "Client not initialized; use `async with`"
        payload = VLLMEmbeddingRequest(input=texts).model_dump()
        resp = await self._client.post(f"{self.base_url}/embeddings", json=payload)
        resp.raise_for_status()
        parsed = VLLMEmbeddingResponse(**resp.json())
        return [item.embedding for item in parsed.data]
