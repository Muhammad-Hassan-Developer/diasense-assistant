# from fastapi import FastAPI
# from pydantic import BaseModel
# from src.chains import run_chat   # 👈 import your function

# app = FastAPI(title="Diasense RAG API")

# class ChatRequest(BaseModel):
#     question: str

# @app.post("/chat")
# def chat(request: ChatRequest):
#     print(f"Received question: {request.question}")
#     # Call your function to process the question
#     result = run_chat(request.question)
#     return {
#         "question": request.question,
#         "answer": result["answer"]
#     }
# # from fastapi import FastAPI
# # from pydantic import BaseModel

# # app = FastAPI(title="Diasense RAG API")

# # class ChatRequest(BaseModel):
# #     question: str

# # @app.post("/chat")
# # def chat(request: ChatRequest):
# #     print(f"Received question: {request.question}")
    
# #     return {
# #         "message": "Question received successfully",
# #         "question": request.question
# #     }
# # uvicorn src.api.chat:app --reload
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