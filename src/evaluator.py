import asyncio
import math
import logging
from typing import Dict, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from src.embeddings import OpenAIEmbedding
from src.config import Config
config=Config()
# Jahan ChatOpenAI define hai:
llm = ChatOpenAI(model=config.open_ai_llm_model, max_tokens=2000)
embeddings=OpenAIEmbedding(model=config.open_ai_llm_model,api_key=config.open_ai_api)
class RagasEvaluator:
    def __init__(self, llm=None, embeddings=None):
        self.metrics = [faithfulness, answer_relevancy]
        # LLM setup - Ensure max_tokens is enough
        self.llm = llm
        self.emb = embeddings

    def _parse_context(self, context: str) -> List[str]:
        if not context:
            return ["No context available"]
        return [context] if isinstance(context, str) else context

    def _safe_float(self, val):
        """NaN values ko JSON compliant banane ke liye helper"""
        try:
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return float(val)
        except:
            return 0.0

    async def evaluate_chat(self, query: str, context: str, response: str) -> Dict:
        try:
            sample = {
                "user_input": [query],
                "retrieved_contexts": [self._parse_context(context)],
                "response": [response],
                "reference": [response] 
            }
            dataset = Dataset.from_dict(sample)

            # RAGAS call in thread
            result = await asyncio.to_thread(
                evaluate,
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.emb
            )

            scores_dict = result.scores[0]

            # SAFE FLOAT conversion taake NaN error na aaye
            f_score = self._safe_float(scores_dict.get("faithfulness"))
            r_score = self._safe_float(scores_dict.get("answer_relevancy"))

            return {
                "averageScore": round((f_score + r_score) / 2, 4),
                "faithfulness": round(f_score, 4),
                "answerRelevancy": round(r_score, 4),
                "contextPrecision": 0.0,
                "contextRecall": 0.0 
            }

        except Exception as e:
            logging.error(f"Ragas Error: {str(e)}")
            # Fallback response taake frontend crash na ho
            return {
                "averageScore": 0.0,
                "faithfulness": 0.0,
                "answerRelevancy": 0.0,
                "error": str(e)
            }