from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.evaluator import RagasEvaluator
from langchain_openai import ChatOpenAI
from src.config import Config
config=Config()
# Jahan ChatOpenAI define hai:
llm = ChatOpenAI(model=config.open_ai_llm_model, max_tokens=2000)
from langchain_openai import OpenAIEmbeddings

# Initialize aise karein
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", # Ya config se uthain
    api_key=config.open_ai_api
)
# Global instance initialize krna taake bar bar LLM load na ho
evaluator = RagasEvaluator(llm=llm,embeddings=embeddings)

router = APIRouter(
    tags=["RAGAS Evaluation"]
)

class RagasRequest(BaseModel):
    query: str
    response: str
    context: str

@router.post("/ragas")
async def evaluate_ragas_endpoint(request: RagasRequest):
    try:
        # Actually calling your evaluator logic
        scores = await evaluator.evaluate_chat(
            query=request.query,
            context=request.context,
            response=request.response
        )
        
        return {
            "status": "success",
            **scores  # This will unpack averageScore, faithfulness, etc.
        }
        
    except Exception as e:
        import logging
        logging.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAGAS evaluation error: {str(e)}")