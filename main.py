# Run the application with: uvicorn main:app --reload
# Ensure to have the necessary dependencies installed:
# pip install fastapi uvicorn transformers faiss-cpu sentence-transformers  pydantic
# and have a FAISS index ready for use.
# The application can be tested using tools like Postman or curl.

# Import necessary libraries.
from fastapi import FastAPI, Query
from app.routers.search_router import router as search_router

# Initialize FastAPI app.
app = FastAPI(
    title="AI Cross lingual documents retrieval app.",
    description="A FastAPI application for retrieving cross-lingual documents using Transformers and FAISS.",
    version="1.0.0"
);

# Include the search router.
app.include_router(search_router, prefix="/api", tags=["search"])

# Define a root endpoint.
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Cross lingual documents retrieval app!"}

# Define a health check endpoint.
@app.get("/health")
async def health_check():
    return {"status": "ok"}