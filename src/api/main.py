from __future__ import annotations
import logging
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from anyio import to_thread

from src.chains import query_chain

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diasense.api")

app = FastAPI(
    title="DiaSense Healthcare AI",
    description="Agentic RAG for Diabetes Analysis",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, example="What are the early signs of diabetes?")

def _serialize_docs(docs: Any) -> List[Dict[str, Any]]:
    if not docs: return []
    return [
        {
            "content": getattr(d, "page_content", str(d)),
            "metadata": getattr(d, "metadata", {})
        } for d in docs
    ]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "OpenAI/Hybrid-RAG"}

@app.post("/query", status_code=status.HTTP_200_OK)
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Processing query: {request.question}")

        # Run heavy RAG logic in a separate thread to keep API responsive
        result = await to_thread.run_sync(query_chain, request.question)

        return {
            "answer": result.get("answer", "I couldn't generate an answer."),
            "documents": _serialize_docs(result.get("documents")),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Pipeline Error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the healthcare query."
        )