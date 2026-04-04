# src/chains.py
import asyncio
from langchain_core.runnables import RunnableParallel, RunnableLambda
from src.config import Config
from src.embeddings import OpenAIEmbedding
from src.vector_store import VectorStore
from src.retrieval import RetrievalManager
from src.sementic_retrieval import SementicRetrieval
from src.bm25_retrieval import BM25Retrieval
from src.runnables import AppRunnables
from src.llms import OpenAILLM

# 1. Setup Configuration & Models
config = Config()
embedding_model = OpenAIEmbedding(
    model=config.open_ai_embedding_model, 
    api_key=config.open_ai_api
)
vectorstore = VectorStore.get_vectorstore(
    collection_name=config.chroma_collection,
    api_key=config.chroma_api_key,
    tenant=config.chroma_tenant,
    database=config.chroma_db,
    embedding_model=embedding_model
)

# 2. Setup Retrieval Components
sr = SementicRetrieval(vectorstore, k=5)
br = BM25Retrieval(chunk_path="data/processed/chunks_700_100.pkl", k=5)
retrieval_mgr = RetrievalManager(sementic_retrieval=sr, bm25_retrieval=br)

app_runnables = AppRunnables(retrieval_mgr)
semantic_retrieval_chain = app_runnables.semantic_retrieval_chain()
bm25_retrieval_chain = app_runnables.bm25_retrieval_chain()

# 3. Define the Hybrid Logic
def merge_documents(results):
    semantic_docs = results.get("semantic", [])
    bm25_docs = results.get("bm25", [])
    combined = semantic_docs + bm25_docs
    unique_docs = []
    seen_content = set()
    for doc in combined:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
    return unique_docs

hybrid_retrieval_chain = (
    RunnableParallel({
        "semantic": semantic_retrieval_chain,
        "bm25": bm25_retrieval_chain
    })
    | RunnableLambda(merge_documents)
)

# 4. Prompt Templates (Initialized at Top)
SYSTEM_PROMPT_TEMPLATE = (
    "You are DiaSense AI, a medical assistant specializing in diabetes. "
    "Use the provided context from 'Standards of Care 2026' to answer accurately. "
    "Always cite the Page numbers provided in the context."
)

HUMAN_PROMPT_TEMPLATE = """
I have retrieved the following information from the medical guidelines:

CONTEXT:
{context}

USER QUESTION: 
{query}

Please provide a detailed answer based ONLY on the context above.
"""

# 5. Async Main Execution Block
# async def rag_chain():
#     query = "What is diabetes?"
#     print(f"\n--- Processing: {query} ---")
    
#     # CRITICAL FIX: We MUST 'await' the async invoke
#     print("🔍 Running Hybrid Retrieval...")
#     docs = await hybrid_retrieval_chain.ainvoke(query)
#     print(f"✅ Retrieved {len(docs)} unique chunks.")
    
#     # Format the context string
#     context_text = "\n\n".join([
#         f"--- Excerpt (Page {d.metadata.get('page_label', 'N/A')}) ---\n{d.page_content}" 
#         for d in docs
#     ])

#     # Initialize the human prompt
#     final_human_prompt = HUMAN_PROMPT_TEMPLATE.format(
#         context=context_text, 
#         query=query
#     )

#     # Initialize and call LLM
#     llm = OpenAILLM(api_key=config.open_ai_api, model=config.open_ai_llm_model)
    
#     print("🤖 Sending to LLM...")
#     # CRITICAL FIX: Also must 'await' the LLM call
#     response = await llm.invoke(
#         system_prompt=SYSTEM_PROMPT_TEMPLATE, 
#         user_prompt=final_human_prompt
#     )
# src/chains.py
import asyncio
from src.llms import OpenAILLM
# ... (all your existing imports)

# INITIALIZE ONCE (Global Scope)
# This prevents the 1.83s "Startup" cost on every API call
# ... (your existing embedding, vectorstore, and retrieval_mgr setup)
llm = OpenAILLM(api_key=config.open_ai_api, model=config.open_ai_llm_model)

async def query_chain(question: str):
    """The main function FastAPI will call"""
    # 1. Retrieval
    docs = await hybrid_retrieval_chain.ainvoke(question)
    
    # 2. Format
    context_text = "\n\n".join([
        f"--- Excerpt (Page {d.metadata.get('page_label', 'N/A')}) ---\n{d.page_content}" 
        for d in docs
    ])

    final_human_prompt = HUMAN_PROMPT_TEMPLATE.format(context=context_text, query=question)

    # 3. LLM Generation
    answer = await llm.invoke(
        system_prompt=SYSTEM_PROMPT_TEMPLATE, 
        user_prompt=final_human_prompt
    )

    return {
        "answer": answer,
        "documents": docs
    }