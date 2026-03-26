from langchain_core.runnables import RunnableLambda

from src.config import Config
from src.vector_store import VectorStore
from src.embeddings import OpenAIEmbedding


class Runnables:

    def __init__(self):
        self.config = Config()

        # Initialize vector store
        self.vs = VectorStore()

        # Initialize embedding model
        self.embedding_model = OpenAIEmbedding(
            model=self.config.open_ai_embedding_model,
            api_key=self.config.open_ai_api
        )

        # Get vectorstore instance
        self.vectorstore = self.vs.get_vectorstore(
            api_key=self.config.chroma_api_key,
            collection_name=self.config.chroma_collection,
            tenant=self.config.chroma_tenant,
            database=self.config.chroma_db,
            embedding_model=self.embedding_model
        )

    def sementic_retrieval_runnable(self):

        def sementic_retrieval(question: str):
            return self.vectorstore.similarity_search(question,k=5)

        return RunnableLambda(sementic_retrieval)
    # 1. Define the logic as a function
    def combine_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # 2. Wrap it in a RunnableLambda
    combine_docs_runnable = RunnableLambda(combine_docs)

    # 3. Use it in a chain
    # Example: retrieval_chain = retriever | combine_docs_runnable | prompt | model