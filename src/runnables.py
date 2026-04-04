from langchain_core.runnables import RunnableLambda
import asyncio
class AppRunnables:
    def __init__(self, retrieval_mgr):
        # Hum poora manager pass kar rahe hain taake flexibility rahe
        self.rm = retrieval_mgr

    def semantic_retrieval_chain(self):
        """
        Ye method 'What is diabetes?' jaise string input leta hai 
        aur documents return karta hai.
        """
        return RunnableLambda(lambda query: asyncio.run(self.rm.get_semantic_docs(query)))
    def bm25_retrieval_chain(self):
        """
        Ye method 'What is diabetes?' jaise string input leta hai 
        aur documents return karta hai.
        """
        return RunnableLambda(lambda query: asyncio.run(self.rm.get_bm25_docs(query)))

    def hybrid_retrieval_chain(self):
        """
        Hybrid retrieval ke liye (Async method ko sync wrapper mein lapetna parta hai)
        """
        import asyncio
        return RunnableLambda(lambda query: asyncio.run(self.rm.get_bm25_docs(query)))
    def rerank_docs_chain(self):
        """
        Input format: {"query": "...", "documents": [...]}
        Output: Top reranked documents with metadata intact.
        """
        def rerank_logic(input_data):
            query = input_data["query"]
            docs = input_data["documents"]
            
            if not docs:
                return []
                
            # LangChain built-in method jo humne RerankManager mein compressor rakha hai
            return self.reranker.compress_documents(documents=docs, query=query)

        return RunnableLambda(rerank_logic)