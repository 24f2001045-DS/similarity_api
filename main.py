from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# Enable CORS (allow all origins, headers, methods)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body structure
class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.post("/similarity")
def similarity(req: SimilarityRequest):
    texts = req.docs + [req.query]

    # Generate embeddings
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [np.array(e.embedding) for e in response.data]

    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Compute similarity scores
    scores = [cosine_similarity(query_embedding, d) for d in doc_embeddings]

    # Get top 3 matches
    top_indices = np.argsort(scores)[-3:][::-1]

    matches = [req.docs[i] for i in top_indices]

    return {"matches": matches}
