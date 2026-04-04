# 
import pickle
from langchain_community.retrievers import BM25Retriever

class BM25Retrieval:
    def __init__(self, chunk_path: str, k: int):
        self.chunk_path = chunk_path
        self.k = k
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        """Sirf aik baar startup par chunks load honge"""
        try:
            with open(self.chunk_path, "rb") as f:
                chunks = pickle.load(f)
            
            # Index yahan aik hi baar ban jaye ga
            retriever = BM25Retriever.from_documents(chunks)
            retriever.k = self.k
            print(f"✅ BM25 Ready: {len(chunks)} chunks loaded.")
            return retriever
        except Exception as e:
            print(f"❌ BM25 Load Error: {e}")
            return None

    def search(self, query: str):
        """Simple and fast search"""
        if not self.retriever:
            return []
        return self.retriever.invoke(query)