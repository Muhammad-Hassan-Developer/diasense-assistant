# src/vectorstore.py

from src.runables import RunnableManager
from src.embeddings import Embeddings
import chromadb
from langchain_chroma import Chroma
from src.config import Config

config = Config()
rm = RunnableManager()
emb = Embeddings()



import chromadb
from langchain_chroma import Chroma

class VectorStore:
    # 1. Added 'self' (or @staticmethod) so the class can call its own methods
    @staticmethod
    def get_chroma_client(api_key, tenant, database):
        return chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )

    @classmethod
    def get_vectorstore(cls, collection_name: str, api_key, tenant, database):
        # 2. Use 'cls' to call the sibling method
        client = cls.get_chroma_client(api_key, tenant, database)
        embedding_model = emb.embedding_model 

        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )


