from src.config import Config
config=Config()
from src.embeddings import OpenAIEmbedding
from src.vector_store import VectorStore
vs=VectorStore()
embedding_model = OpenAIEmbedding(model=config.open_ai_embedding_model, api_key=config.open_ai_api)
vectorstore = vs.get_vectorstore(api_key=config.chroma_api_key,collection_name=config.chroma_collection, tenant=config.chroma_tenant, database=config.chroma_db,embedding_model=embedding_model)
def get_diabetes_retriever(query):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(query)   # 👈 yahan question diya

    for doc in docs:
        print(doc.page_content)

    return docs
get_diabetes_retriever("What is diabetes?")
# python -m src.retrieval