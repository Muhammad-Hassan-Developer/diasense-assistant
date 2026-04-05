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


import asyncio
from src.llms import OpenAILLM
# ... (all your existing imports)

# # INITIALIZE ONCE (Global Scope)
# # This prevents the 1.83s "Startup" cost on every API call
# # ... (your existing embedding, vectorstore, and retrieval_mgr setup)
# llm = OpenAILLM(api_key=config.open_ai_api, model=config.open_ai_llm_model)

# async def query_chain(question: str):
#     """The main function FastAPI will call"""
#     # 1. Retrieval
#     docs = await hybrid_retrieval_chain.ainvoke(question)
    
#     # 2. Format
#     context_text = "\n\n".join([
#         f"--- Excerpt (Page {d.metadata.get('page_label', 'N/A')}) ---\n{d.page_content}" 
#         for d in docs
#     ])

#     final_human_prompt = HUMAN_PROMPT_TEMPLATE.format(context=context_text, query=question)

#     # 3. LLM Generation
#     answer = await llm.invoke(
#         system_prompt=SYSTEM_PROMPT_TEMPLATE, 
#         user_prompt=final_human_prompt
#     )

#     return {
#         "answer": answer,
#         "documents": docs
#     }
import time
import asyncio
from src.llms import OpenAILLM
# ... (all your other imports from config, vector_store, etc.)

# Pricing dictionary (optional but helpful for dashboard)
PRICING = {
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000}
}

# 1. Global Initialization
llm = OpenAILLM(api_key=config.open_ai_api, model=config.open_ai_llm_model)

async def query_chain(question: str):
    """The main function with detailed performance and source tracking"""
    
    start_total = time.perf_counter() # Total latency start
    
    # --- STEP 1: RETRIEVAL ---
    start_retrieval = time.perf_counter()
    docs = await hybrid_retrieval_chain.ainvoke(question)
    retrieval_time = round(time.perf_counter() - start_retrieval, 4)
    
    # --- STEP 2: FORMAT CONTEXT ---
    # Hum context build kar rahe hain aur sath hi sources ki list bhi
    context_parts = []
    sources = []
    
    for d in docs:
        page = d.metadata.get('page_label', 'N/A')
        content = d.page_content
        context_parts.append(f"--- Excerpt (Page {page}) ---\n{content}")
        
        # Dashboard ke liye clean sources object
        sources.append({
            "page": page,
            "snippet": content[:150] + "...", # Preview for frontend
            "metadata": d.metadata
        })

    context_text = "\n\n".join(context_parts)
    final_human_prompt = HUMAN_PROMPT_TEMPLATE.format(context=context_text, query=question)

    # --- STEP 3: LLM GENERATION ---
    start_llm = time.perf_counter()
    
    # Note: Make sure your invoke() returns the full response object to get usage
    # If it only returns string, you might need: response_obj = await llm.client.chat.completions.create(...)
    full_response = await llm.invoke(
        system_prompt=SYSTEM_PROMPT_TEMPLATE, 
        user_prompt=final_human_prompt
    )
    
    llm_time = round(time.perf_counter() - start_llm, 4)
    total_latency = round(time.perf_counter() - start_total, 4)

    # --- STEP 4: TOKEN & COST CALCULATION ---
    # Assuming full_response has usage (Standard OpenAI object)
    # If your llm.invoke only returns a string, you will need to adjust your llm class
    usage = getattr(full_response, 'usage', None)
    
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    
    # Simple cost estimate
    model_name = config.open_ai_llm_model
    rates = PRICING.get(model_name, PRICING["gpt-4o-mini"])
    estimated_cost = (input_tokens * rates["input"]) + (output_tokens * rates["output"])

    # --- FINAL OUTPUT ---
    return {
        "answer": full_response.choices[0].message.content if usage else full_response,
        "context_used": context_text,
        "performance": {
            "total_latency": f"{total_latency}s",
            "retrieval_latency": f"{retrieval_time}s",
            "llm_latency": f"{llm_time}s"
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6)
        },
        "sources": sources
    }