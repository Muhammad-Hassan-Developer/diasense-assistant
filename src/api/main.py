from fastapi import FastAPI
from src.api import qa

app = FastAPI(title="Diasense RAG API")

# Include routers
app.include_router(qa.router, prefix="/qa", tags=["QA"])
@app.get("/")
def root():
    return {"message": "Welcome to Diasense RAG API"}