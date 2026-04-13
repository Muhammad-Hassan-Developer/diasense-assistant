import asyncio
from typing import Dict, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
# Baki jo metrics aap use kar rahe hain unhein yahan add karein

class RagasEvaluator:
    def __init__(self, llm=None, embeddings=None):
        # Initializing metrics
        self.metrics = [faithfulness, answer_relevancy]
        self.llm = llm
        self.emb = embeddings

    def _parse_context(self, context: str) -> List[str]:
        """
        RAGAS expect karta hai ke context aik list of strings ho.
        Ye function raw string ko sahi format mein convert karta hai.
        """
        if not context:
            return ["No context available"]
        
        # Agar context string hai to usay list mein wrap karein
        if isinstance(context, str):
            return [context]
        
        return context

    async def evaluate_chat(self, query: str, context: str, response: str) -> Dict:
        """
        Main evaluation function jo asyncio compatibility ke sath chalta hai.
        """
        try:
            # 1. Dataset prepare karein (RAGAS format)
            sample = {
                "user_input": [query],
                "retrieved_contexts": [self._parse_context(context)],
                "response": [response],
                "reference": [response] # Reference ko response ke barabar rakh rahe hain for now
            }
            dataset = Dataset.from_dict(sample)

            # 2. RAGAS evaluation ko alag thread mein chalana (Render/Linux fix)
            result = await asyncio.to_thread(
                evaluate,
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.emb
            )

            # 3. Result se scores nikalna
            # Ragas result.scores aik list return karta hai (index 0 for our single row)
            scores_dict = result.scores[0]

            f_score = float(scores_dict.get("faithfulness", 0))
            r_score = float(scores_dict.get("answer_relevancy", 0))

            # 4. Final response taiyar karein
            return {
                "averageScore": round((f_score + r_score) / 2, 4),
                "faithfulness": round(f_score, 4),
                "answerRelevancy": round(r_score, 4),
                "contextPrecision": 0.0,
                "contextRecall": 0.0 
            }

        except Exception as e:
            import logging
            logging.error(f"Ragas Evaluation Process Failed: {str(e)}")
            # Error throw karein taake FastAPI main route handle kar sakay
            raise Exception(f"RAGAS evaluation error: {str(e)}")