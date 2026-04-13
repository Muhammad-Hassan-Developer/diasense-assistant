from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import logging
import time
from src.api.routes import ragas,qa


# Import the updated async function from your chains file




app = FastAPI(title="DiaSense Healthcare AI")

# Enable CORS for Frontend connectivity (v0.dev, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ragas.router)
app.include_router(qa.router)
# 1. Health Check
@app.get("/health")
async def health():
    return {"status": "healthy"}

# 2. Updated Main RAG Query Endpoint

# async def query_endpoint(request: QueryRequest):
#     try:
#         logger.info(f"Processing Query: {request.query}")

#         # Calling the updated RAG pipeline
#         # result ab aik structured dict return karega
#         result = await query_chain(request.query)

#         # Mapping data to match New Frontend Expectations (including metrics)
#         return {
#             "status": "success",
#             "answer": result["answer"],
#             "sources": result["sources"],
#             "context":result["context_used"], # Direct source objects (page, snippet)
#             "metrics": {
#                 "performance": result["performance"], # Latency (total, retrieval, llm)
#                 "usage": result["usage"]             # Tokens and Cost
#             },
#             # "context_raw": result["context_used"] # Optional: Agar debug karna ho
#         }

#     except Exception as e:
#         logger.error(f"RAG Pipeline Error: {str(e)}")
#         # Production mein 'str(e)' ki jagah generic message dein
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# # Run command for production:
# # uvicorn src.api.main:app --host 0.0.0.0 --port 8000