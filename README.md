# Essay RAG Sentence Retriever

## Overview
- Retrieves the most relevant sentences from a PDF book for a given student essay using cosine similarity.
- Uses a local Chroma vector store with persistence and a vLLM-compatible embeddings HTTP API.
- Returns sentence, score, context (previous/next sentence), and location (chapter, page).

## Prerequisites
- Python 3.8+
- A running embeddings server compatible with the OpenAI-style `/v1/embeddings` endpoint (e.g., vLLM with `intfloat/e5-small`). See [vllm](https://docs.vllm.ai/en/latest/getting_started/installation/).
- If GPU is not accessible, you can use STAPI (Sentence Transformers API):
```
# Clone the repository (or install from source)
git clone https://github.com/substratusai/stapi.git
cd stapi

# Install dependencies (requires Python 3.9, 3.10, or 3.11)
pip install -r requirements.txt


# Set the environment variable to use the E5 model
export MODEL="intfloat/e5-small-v2"

# Start the server using Uvicorn on the specified port
# The default endpoint will be http://localhost:8000/v1/embeddings
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Installation
- Create and activate a virtual environment:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
- Install dependencies:
  - `pip install -r requirements.txt`

## Configure Embeddings Server
- Default base URL: `http://localhost:8000/v1`.

## Index and Retrieve
- Ensure the embeddings server is running.
- Run: `python demo.py`
- This repository uses Chroma in-memory (ephemeral) mode; indices live only during the process.

## Future Work
- **Synthetic Dataset Generation and Hyper-parameter Tuning**  
  Produce a high-quality synthetic dataset via a language model and use it to systematically optimize retrieval parameters (e.g., sentences per query, total sentences per essay, similarity threshold).  
  A preliminary dataset is available at `test/synthetic_llm_30.json`.
