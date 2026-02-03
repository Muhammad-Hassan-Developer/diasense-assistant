import os
from dotenv import load_dotenv

# .env file ko load karein
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("Groq_api") # Jo naam aapne .env mein rakha hai
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL = "mixtral-8x7b-32768"
    DB_DIR = "db"