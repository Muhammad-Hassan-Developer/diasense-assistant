from fastapi import APIRouter
from pydantic import BaseModel
from src.retrieval import query_docs  

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/")
async def send_query(request: QueryRequest):
    # Call the real retrieval + OpenAI function
    result = query_docs(request.query)
    return result