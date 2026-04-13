from typing import Dict, List, Union
from ragas import evaluate, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset # Import this
from src.config import Config
config=Config()

class RagasEvaluator:
    def __init__(self):
        config = Config()
        llm = ChatOpenAI(
            model=config.open_ai_llm_model, 
            api_key=config.open_ai_api,
            temperature=0
        )
        emb = OpenAIEmbeddings(
            model=config.open_ai_embedding_model, 
            api_key=config.open_ai_api
        )
        
        self.llm = LangchainLLMWrapper(llm)
        self.emb = LangchainEmbeddingsWrapper(emb)
        self.metrics = [faithfulness, answer_relevancy]

    def _parse_context(self, context: Union[str, List[str]]) -> List[str]:
        if isinstance(context, list): return context
        for sep in ["---", "\n\n"]:
            if sep in context: return [c.strip() for c in context.split(sep) if c.strip()]
        return [context.strip()]

    async def evaluate_chat(self, query: str, context: str, response: str) -> Dict:
    # 1. Prepare sample with lists
        sample = {
            "user_input": [query],
            "retrieved_contexts": [self._parse_context(context)],
            "response": [response],
            "reference": [response] # Optional: Only if using context_precision
        }
        
        # 2. Convert to Dataset
        dataset = Dataset.from_dict(sample)
        
        # 3. Run evaluate
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.emb
        )
        
        # ERROR FIX: result.scores se direct dictionary nikalna
        # result.scores aik dictionary-like object hota hai
        try:
            # Agar result.scores direct access ho raha hai (Ragas latest)
            scores_dict = result.scores[0] 
        except:
            # Backup: Agar pandas format mein convert karna pare
            scores_dict = result.to_pandas().iloc[0].to_dict()

        # Scores extract krna aur response format ke mutabiq map krna
        faithfulness_val = scores_dict.get("faithfulness", 0)
        relevancy_val = scores_dict.get("answer_relevancy", 0)
        
        # Average calculate karein
        avg = round((faithfulness_val + relevancy_val) / 2, 4)
        
        return {
            "averageScore": avg,
            "faithfulness": round(float(faithfulness_val), 4),
            "answerRelevancy": round(float(relevancy_val), 4),
            "contextPrecision": 0.0, # Placeholder
            "contextRecall": 0.0 
    }