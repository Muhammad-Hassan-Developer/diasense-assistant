import os
from dotenv import load_dotenv

# .env file ko load karein
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API") # Jo naam aapne .env mein rakha hai
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    GROQ_MODEL =os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    chroma_api_key=os.getenv("CHROMA_API_KEY")
    chroma_tenant=os.getenv("CHROMA_TENANT")
    chroma_db=os.getenv("CHROMA_DATABASE")
# "mixtral-8x7b-32768",
    
