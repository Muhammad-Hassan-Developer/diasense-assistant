import pickle
from langchain_community.retrievers import BM25Retriever

class BM25Manager:

    def __init__(self, chunk_path):
        self.chunk_path = chunk_path
        self.retriever = None

    def load_chunks(self):
        with open(self.chunk_path, "rb") as f:
            chunks = pickle.load(f)
        return chunks

    def build_retriever(self, k=5):
        chunks = self.load_chunks()
        self.retriever = BM25Retriever.from_documents(chunks)
        self.retriever.k = k
        print("BM25 retriever ready")

    def query(self, query):
        return self.retriever.invoke(query)