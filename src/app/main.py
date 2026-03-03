
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from src.chains import run_chat

# 1️⃣ FastAPI app
app = FastAPI(title="Diasense Chat API")

# 2️⃣ Request model
class ChatRequest(BaseModel):
    question: str

# 3️⃣ Router (optional, but good for modularity)
router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = run_chat(request.question)
        return result
    except Exception as e:
        return {"error": str(e)}

# 4️⃣ Include router
app.include_router(router)
# uvicorn src.app.main:app --reload