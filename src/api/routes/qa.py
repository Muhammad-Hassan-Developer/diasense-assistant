from fastapi import APIRouter, HTTPException
from pydantic import BaseModel,Field
from src.chains import query_chain
import logging
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diasense.api")
router = APIRouter()
# Request Schema
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
@router.post("/query") 
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"Processing Query: {request.query}")

        # Calling the updated RAG pipeline
        # result ab aik structured dict return karega
        result = await query_chain(request.query)

        # Mapping data to match New Frontend Expectations (including metrics)
        return {
            "status": "success",
            "answer": result["answer"],
            "sources": result["sources"],
            "context":result["context_used"], # Direct source objects (page, snippet)
            "metrics": {
                "performance": result["performance"], # Latency (total, retrieval, llm)
                "usage": result["usage"]             # Tokens and Cost
            },
            # "context_raw": result["context_used"] # Optional: Agar debug karna ho
        }

    except Exception as e:
        logger.error(f"RAG Pipeline Error: {str(e)}")
        # Production mein 'str(e)' ki jagah generic message dein
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
