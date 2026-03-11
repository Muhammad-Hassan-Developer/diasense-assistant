from src.config import Config
config=Config()
from src.embeddings import OpenAIEmbedding
from src.vector_store import VectorStore
vs=VectorStore()

embedding_model = OpenAIEmbedding(model=config.open_ai_embedding_model, api_key=config.open_ai_api)
vectorstore = vs.get_vectorstore(api_key=config.chroma_api_key,collection_name=config.chroma_collection, tenant=config.chroma_tenant, database=config.chroma_db,embedding_model=embedding_model)
retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5})
query = "What is diabetes?"
docs = retriever.invoke(query)

# Check what you got
for doc in docs:
    print(doc.page_content)



docs_text = "\n\n".join([doc.page_content for doc in docs])
from openai import OpenAI
openai_client = OpenAI(api_key=config.open_ai_api).chat.completions.create(model=config.open_ai_llm_model,                                                                            
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant for diabetes related questions. Use the following retrieved documents to answer the question. If you don't know the answer, say you don't know. Use all the retrieved documents to answer the question."
    },
    {
        "role": "user",
        "content": f"{query}\n\nRetrieved Documents:\n{docs_text}"
    }
]
)
print(openai_client.choices[0].message.content)
# python -m src.retrieval