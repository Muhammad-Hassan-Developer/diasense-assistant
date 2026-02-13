from src.config import Config
from langchain_huggingface import HuggingFaceEmbeddings
config=Config()
model_name = config.HUGGINGFACE_EMBEDDING_MODEL
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
class Embeddings:
    def __init__(self):
        self.embedding_model = hf

    def get_embeddings(self, texts):
        """Get embeddings for a list of texts."""
        return self.embedding_model.embed_documents(texts)
    

from src.config import Config
from langchain_huggingface import HuggingFaceEmbeddings

config = Config()

from typing import List

class Embeddings:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.HUGGINGFACE_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    def get_embedding_model(self):
        return self.embedding_model
