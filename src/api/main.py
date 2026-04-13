from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import logging
import time
from src.api.routes import ragas,qa
import asyncio
import sys

# Windows par masla nahi hota, lekin Linux/Render par uvloop ko disable krna prta hai
import sys
import asyncio

# Safe handling for uvloop (Linux only)
if sys.platform != "win32":
    try:
        import uvloop # type: ignore
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except ImportError:
        pass

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