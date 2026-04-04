from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from src.config import Config

config = Config()

class RerankManager:
    def __init__(self, top_n: int = 3):
        # LangChain ka built-in compressor
        self.compressor = CohereRerank(
            cohere_api_key=config.cohere_api_key,
            model=config.cohere_model,
            top_n=top_n
        )

    def get_reranker(self):
        return self.compressor
