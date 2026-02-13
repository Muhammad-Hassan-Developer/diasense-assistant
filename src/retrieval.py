from langchain_core.runnables import RunnableLambda
from src.llms import Llms_client
from src.config import Config
from src.prompts import Prompts
llms_client = Llms_client()
config=Config()
prompts=Prompts()

from src.vector_store import get_vectorstore
from src.embeddings import get_embeddings  # jo tum use kar rahe ho

def get_diabetes_retriever():
    embeddings = get_embeddings()

    vectorstore = get_vectorstore(
        collection_name="diabetes",  # 👈 EXISTING collection
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # top-4 chunks
    )

    return retriever