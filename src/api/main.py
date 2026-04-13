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


# # Run command for production:
# # uvicorn src.api.main:app --host 0.0.0.0 --port 8000