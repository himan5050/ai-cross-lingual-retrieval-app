from fastapi import APIRouter, Query # type: ignore
import pandas as pd
from sentence_transformers import SentenceTransformer # type: ignore
import faiss # type: ignore
import numpy as np # type: ignore
from app.models.search_response import SearchResponse

router = APIRouter()



# Load jsonl file into dataframe
documents = pd.read_json("artifacts/shared/docs__processed_chunked_cleaned.jsonl", lines=True);

# 2. Load model.
model_id = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1';
model = SentenceTransformer(model_id, device='cpu')

# 4. Create FAISS index
index_path = 'artifacts/indexes/xlmr-indexed.faiss'
index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)

@router.get("/search", response_model=list[SearchResponse])
def search(query: str = Query(..., description="Search query in any language")):
    # Encode query
    query_embedding = model.encode([query])

    # normalize query too.
    faiss.normalize_L2(query_embedding)

    # Search in FAISS
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k=20)

    # Collect results
    results = []
    for idx, score in zip(I[0], D[0]):
        doc = documents.iloc[idx]
        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "text": doc["text"],
            "lang": doc["lang"],
            "score": float(score)
        })
    return results