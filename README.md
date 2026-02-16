# Text Similarity API (FastAPI + OpenAI)

Simple FastAPI service that finds the most similar documents to a query using OpenAI embeddings and cosine similarity.

## Features
- Generate embeddings using OpenAI
- Compute cosine similarity with NumPy
- Returns top 3 most similar documents
- FastAPI REST endpoint
- CORS enabled

## Tech Stack
FastAPI • OpenAI API • NumPy • Pydantic • Uvicorn

## Setup

Install dependencies:
pip install fastapi uvicorn openai numpy pydantic

Set OpenAI API key:
export OPENAI_API_KEY=your_key   (Mac/Linux)
set OPENAI_API_KEY=your_key      (Windows)

## Run server
uvicorn main:app --reload

Open:
http://127.0.0.1:8000/docs

## API Usage

POST /similarity

Request body:
{
  "docs": ["text1", "text2", "text3", "text4"],
  "query": "search text"
}

Response:
{
  "matches": ["most similar doc1", "doc2", "doc3"]
}

## Notes
- Uses model: text-embedding-3-small
- Computes similarity using cosine similarity
- Returns top 3 matching documents
