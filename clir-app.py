# Run the application with: uvicorn main:app --reload
# Ensure to have the necessary dependencies installed:
# pip install fastapi uvicorn transformers faiss-cpu sentence-transformers  pydantic
# and have a FAISS index ready for use.
# The application can be tested using tools like Postman or curl.

# Import necessary libraries.
# install first:
# pip install fastapi uvicorn sentence-transformers faiss-cpu mysql-connector-python

from fastapi import FastAPI, Query # type: ignore
from app.routes.search import router as api_search_router

app = FastAPI(
    title="Cross-lingual Semantic Search API",
    description="A FastAPI app for multilingual semantic search using FAISS + MySQL",
    version="1.0.0"
)

# Register Routes
app.include_router(api_search_router, prefix="/api", tags=["Search"])