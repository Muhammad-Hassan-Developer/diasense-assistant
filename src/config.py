import os
from dotenv import load_dotenv

# .env file ko load karein
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API") # Jo naam aapne .env mein rakha hai
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    GROQ_MODEL =os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    # chroma configuration
    chroma_api_key=os.getenv("CHROMA_API_KEY")
    chroma_tenant=os.getenv("CHROMA_TENANT")
    chroma_db=os.getenv("CHROMA_DATABASE")
    chroma_collection=os.getenv("CHROMA_COLLECTION")
    gemini_api=os.getenv("GEMINI_API")
    google_project_id=os.getenv("GOOGLE_PROJECT_ID")
    open_ai_api=os.getenv("OPENAI_API_KEY")
    open_ai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL")
    open_ai_llm_model=os.getenv("OPENAI_LLM_MODEL")
