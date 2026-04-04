import asyncio
from typing import List
from langchain_core.documents import Document

class RetrievalManager:
    def __init__(self, sementic_retrieval, bm25_retrieval, k: int = 5):
        self.sementic_retrieval = sementic_retrieval
        self.bm25_retrieval = bm25_retrieval
        self.k = k  # Fixed: Now k is stored in the instance

    async def get_semantic_docs(self, query: str) -> List[Document]:
        """Returns only Semantic Docs (Vector Search)"""
        # Using ainvoke for non-blocking I/O
        return await self.sementic_retrieval.retriever.ainvoke(query)

    async def get_bm25_docs(self, query: str) -> List[Document]:
        """Returns only BM25 Docs (Keyword Search)"""
        return await self.bm25_retrieval.retriever.ainvoke(query)

    async def get_hybrid_docs(self, query: str) -> List[Document]:
        """Runs both in parallel and merges without duplicates"""
        # 1. Create tasks for parallel execution
        semantic_task = self.get_semantic_docs(query)
        bm25_task = self.get_bm25_docs(query)
        
        # 2. Gather results simultaneously
        results = await asyncio.gather(semantic_task, bm25_task)
        
        # 3. Flatten, Deduplicate, and Slice
        all_docs = results[0] + results[1]
        unique_docs = []
        seen_content = set()

        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        # Return only the top 'k' results
        return unique_docs[:self.k]