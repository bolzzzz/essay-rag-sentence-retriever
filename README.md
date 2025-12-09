# Essay RAG Sentence Retriever

## Overview
- Retrieves the most relevant sentences from a PDF book for a given student essay using cosine similarity.
- Uses a local Chroma vector store with persistence and a vLLM-compatible embeddings HTTP API.
- Returns sentence, score, context (previous/next sentence), and location (chapter, page).

## Prerequisites
- Python 3.8+
- A running embeddings server compatible with the OpenAI-style `/v1/embeddings` endpoint (e.g., vLLM with `intfloat/e5-small`).
- Book file, e.g., `sample_book.pdf`.

## Installation
- Create and activate a virtual environment:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
- Install dependencies:
  - `pip install -r requirements.txt`

## Configure Embeddings Server
- Default base URL: `http://localhost:8000/v1`.
- If your server runs elsewhere, change `base_url` when instantiating `EmbeddingClient` or set an environment variable and pass it through.

## Index and Retrieve
- Ensure the embeddings server is running.
- Run: `python demo.py`
- This repository uses Chroma in-memory (ephemeral) mode; indices live only during the process.

## Future Work
- **Synthetic Dataset Generation and Hyper-parameter Tuning**  
  Produce a high-quality synthetic dataset via a language model and use it to systematically optimize retrieval parameters (e.g., sentences per query, total sentences per essay, similarity threshold).  
  A preliminary dataset is available at `test/synthetic_llm_30.json`.
