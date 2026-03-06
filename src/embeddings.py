from src.config import Config
from langchain_huggingface import HuggingFaceEmbeddings
config=Config()
model_name = config.HUGGINGFACE_EMBEDDING_MODEL
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
from google import genai

client = genai.Client()

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents= [
            "What is the meaning of life?",
            "What is the purpose of existence?",
            "How do I bake a cake?"
        ]
)

for embedding in result.embeddings:
    print(embedding)
class Embeddings_client:


