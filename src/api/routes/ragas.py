from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.evaluator import RagasEvaluator

# Global instance initialize krna taake bar bar LLM load na ho
evaluator = RagasEvaluator()

router = APIRouter(
    prefix="/evaluate",
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