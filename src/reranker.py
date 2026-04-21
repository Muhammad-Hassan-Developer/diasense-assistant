# src/reranker.py
from langchain_cohere import CohereRerank
from src.config import Config

config = Config()

class RerankManager:
    def __init__(self, top_n):
        """
        Cohere Reranker initialization.
        """
        self.reranker = CohereRerank(
            cohere_api_key=config.cohere_api_key,
            model=config.cohere_model, # e.g., 'rerank-v4.0-pro'
            top_n=top_n
        )

    def rerank_documents(self, query: str, documents: list):
        """
        Manual reranking for a list of documents.
        """
        if not documents:
            return []
        
        # LangChain ka compressor use karte hue documents ko rerank karna
        return self.reranker.compress_documents(documents, query)