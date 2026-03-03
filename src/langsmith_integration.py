import os
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

def setup_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

    client = Client()

    return client