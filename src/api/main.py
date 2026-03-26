from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chains import query_chain


app = FastAPI(title="Diasense RAG API")

logger = logging.getLogger("diasense.api")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer")


def _serialize_documents(docs: Any) -> List[Dict[str, Any]]:
    """
    Convert LangChain `Document` objects (or compatible shapes) into JSON-safe dicts:
      page_content -> content
      metadata -> metadata
    """
    if docs is None:
        return []

    if not isinstance(docs, list):
        docs = list(docs)

    out: List[Dict[str, Any]] = []
    for d in docs:
        # LangChain Document: has `.page_content` and `.metadata`
        content = getattr(d, "page_content", None)
        metadata = getattr(d, "metadata", None)

        # Fallback: already a dict-like object
        if content is None and isinstance(d, dict):
            content = d.get("page_content") or d.get("content")
            metadata = d.get("metadata", {})

        out.append(
            {
                "content": content if isinstance(content, str) else ("" if content is None else str(content)),
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )

    return out


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(request: QueryRequest) -> Dict[str, Any]:
    try:
        logger.info("Request received: /query")
        logger.info("Question: %s", request.question)

        logger.info("RAG chain starting")
        try:
            from anyio import to_thread

            result = await to_thread.run_sync(query_chain, request.question)
        except Exception:
            # Fallback if anyio isn't available or thread execution fails.
            result = query_chain(request.question)

        answer = result.get("answer") if isinstance(result, dict) else None
        documents = result.get("documents") if isinstance(result, dict) else None

        response = {
            "answer": "" if answer is None else str(answer),
            "documents": _serialize_documents(documents),
        }
        logger.info("RAG chain complete (docs=%s)", len(response["documents"]))
        return response
    except Exception as e:
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=500, detail=str(e)) from e