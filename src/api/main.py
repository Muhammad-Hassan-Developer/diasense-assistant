# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware  # Corrected Import
# from pydantic import BaseModel, Field
# import logging

# # Import the async function from your chains file
# from src.chains import query_chain

# # Logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("diasense.api")

# app = FastAPI(title="DiaSense Healthcare AI")

# # Enable CORS for Frontend connectivity
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows v0.dev and localhost
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request Schema matching Frontend call
# class QueryRequest(BaseModel):
#     query: str = Field(..., min_length=3, max_length=500)

# # 1. Health Check Endpoint (Fixes "API Unavailable" error)
# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "service": "DiaSense AI"}

# # 2. Main RAG Query Endpoint
# @app.post("/query")
# async def query_endpoint(request: QueryRequest):
#     try:
#         logger.info(f"User Query received: {request.query}")

#         # Calling the async RAG pipeline
#         result = await query_chain(request.query)

#         # Mapping data to match Frontend Expectations
#         return {
#             "answer": result["answer"],
#             "source_nodes": [
#                 {
#                     "content": d.page_content,
#                     "page": d.metadata.get("page_label", "N/A"),
#                     "source": d.metadata.get("source", "Medical Guideline")
#                 } for d in result["documents"]
#             ],
#             "status": "success"
#         }

#     except Exception as e:
#         logger.error(f"RAG Pipeline Error: {str(e)}")
#         # Returning exact error for debugging (change to generic for production)
#         raise HTTPException(status_code=500, detail=str(e))

# # Deployment Note: Render uses 'PORT' env variable, usually 10000
# # Run command: uvicorn main:app --host 0.0.0.0 --port 10000
# # Run with: uvicorn main:app --reload
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import time

# Import the updated async function from your chains file
from src.chains import query_chain

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diasense.api")

app = FastAPI(title="DiaSense Healthcare AI")

# Enable CORS for Frontend connectivity (v0.dev, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Schema
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)

# 1. Health Check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "DiaSense AI"}

# 2. Updated Main RAG Query Endpoint
@app.post("/query")
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
            "sources": result["sources"], # Direct source objects (page, snippet)
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

# Run command for production:
# uvicorn main:app --host 0.0.0.0 --port 10000