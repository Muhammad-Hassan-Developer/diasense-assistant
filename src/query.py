from src.vector_store import get_vectorstore
from src.embeddings import Embeddings

def query_vectorstore(query: str, collection_name="diabetes_2026_pdf", top_k=5):
    emb = Embeddings()
    vectorstore = get_vectorstore(collection_name=collection_name, embedding_function=emb)
    
    # Perform similarity search for top_k relevant docs
    results = vectorstore.similarity_search(query, k=top_k)
    
    # results is a list of Documents, you can print or process them
    return results

query = "What are diabetes ?"
docs = query_vectorstore(query)
for i, doc in enumerate(docs):
    print(f"Result {i+1}:\n{doc.page_content}\n---\n")
