from fastapi import APIRouter, Query
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pydantic import BaseModel


router = APIRouter()
@router.get("/search", response_model=list[SearchResponse])
def DocumentSearch(query: str = Query(..., description="Search query in any language"), model_name: str = Query("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", description="Model name for embedding")):

    # 1. Load documents.
    # documents = Connector.fetch_documents()

    # Load model
    model = SentenceTransformer(model_name)

    # Enocde query.
    query_embedding = model.encode([query])

    # Load FAISS index
    dimension = 768  # depends on embedding model
    index = faiss.IndexFlatL2(dimension)

    # Search in FAISS
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k=5)

    # Load FAISS index from file (assuming it's saved as 'faiss_index.index')
    # Collect results
    results = []
    for idx, score in zip(I[0], D[0]):
        doc = documents[idx]
        results.append({
            "id": doc["id"],
            "text": doc["text"],
            "score": float(score)
        })
    return results