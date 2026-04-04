# main.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import logging

# Import the async function from your chains file
from src.chains import query_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diasense.api")

app = FastAPI(title="DiaSense Healthcare AI")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        logger.info(f"User Query: {request.question}")

        # Calling our async RAG pipeline directly
        result = await query_chain(request.question)

        return {
            "answer": result["answer"],
            "documents": [
                {
                    "content": d.page_content,
                    "page": d.metadata.get("page_label", "N/A")
                } for d in result["documents"]
            ],
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Run with: uvicorn main:app --reload