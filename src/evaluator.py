import asyncio
from typing import Dict
from datasets import Dataset
from ragas import evaluate

# ... baki imports same ...

class RagasEvaluator:
    # ... __init__ and other methods same ...

    async def evaluate_chat(self, query: str, context: str, response: str) -> Dict:
        # 1. Prepare dataset as usual
        sample = {
            "user_input": [query],
            "retrieved_contexts": [self._parse_context(context)],
            "response": [response],
            "reference": [response] 
        }
        dataset = Dataset.from_dict(sample)

        # 2. ERROR FIX: evaluate ko asyncio.to_thread mein wrap karein
        # Is se Ragas ka nested loop main loop se alag ho jayega
        try:
            result = await asyncio.to_thread(
                evaluate,
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.emb
            )
            
            # Scores extract karein
            scores_dict = result.scores[0]
            
            f_score = float(scores_dict.get("faithfulness", 0))
            r_score = float(scores_dict.get("answer_relevancy", 0))

            return {
                "averageScore": round((f_score + r_score) / 2, 4),
                "faithfulness": round(f_score, 4),
                "answerRelevancy": round(r_score, 4),
                "contextPrecision": 0.0,
                "contextRecall": 0.0 
            }
        except Exception as e:
            print(f"Internal RAGAS Error: {e}")
            raise e