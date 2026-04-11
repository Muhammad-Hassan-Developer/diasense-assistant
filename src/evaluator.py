from typing import Dict, List, Union
from ragas import evaluate, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class RagasEvaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        # LLM aur Embeddings ko wrap krna RAGAS ke liye
        llm = ChatOpenAI(model=model, temperature=0)
        emb = OpenAIEmbeddings(model="text-embedding-3-small")
        
        self.llm = LangchainLLMWrapper(llm)
        self.emb = LangchainEmbeddingsWrapper(emb)
        self.metrics = [faithfulness, answer_relevancy, context_precision]

    def _parse_context(self, context: Union[str, List[str]]) -> List[str]:
        """Context ko list of strings mein convert krna."""
        if isinstance(context, list): return context
        # Agar string hai to split kro, warna single item list
        for sep in ["---", "\n\n"]:
            if sep in context: return [c.strip() for c in context.split(sep) if c.strip()]
        return [context.strip()]

    def evaluate_chat(self, query: str, context: Union[str, List[str]], response: str) -> Dict:
        """Single chat evaluate kr ke dictionary return krta hai."""
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=self._parse_context(context),
            response=response
        )
        
        result = evaluate(
            dataset=[sample],
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.emb
        )
        
        # Scores extract krna aur round off krna
        scores = result.scores[0]
        scores = {k: round(float(v), 4) for k, v in scores.items()}
        scores["overall_score"] = round(sum(scores.values()) / len(scores), 4)
        
        return scores