from src.runables import RunnableManager
from src.vector_store import VectorStore
from src.config import Config
from src.embeddings import Embeddings
from src.loader import Loader
from langchain_community.document_loaders import TextLoader
from src.prompts import Prompts
prompts=Prompts()
# 4. Initialize components
loader=Loader()
rm = RunnableManager()
config = Config()
emb = Embeddings()
# initialinze the vector stores
vs = VectorStore()
diasense_vs=vs.get_vectorstore(
    collection_name="diabetes_2026_pdf",
    api_key=config.chroma_api_key,
    tenant=config.chroma_tenant,
    database=config.chroma_db
)

# # 5. Create retriever and perform search
retriever_docs = diasense_vs.as_retriever(search_type="similarity", search_kwargs={"k": 10})


rm=RunnableManager()
all_prompts = loader.load_from_dir("prompts/",glob="**/*.text",loader_cls=TextLoader)
question_prompt = all_prompts[2].page_content
system_prompt = all_prompts[1].page_content
human_prompt = all_prompts[0].page_content
print(human_prompt)

from src.llms import Llms_client
llm_client = Llms_client()

def run_chat(question: str):
    # 1️⃣ Retrieve documents
    retrieval = rm.retrieval_prompt_runnable()  # optional: pass prompt later if needed
    chain = retrieval | retriever_docs
    docs = chain.invoke(question)

    # 2️⃣ Build context
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"Retrieved {len(docs)} documents for question: '{question}'")
    print(f"Context: {context}")

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt)

    ])
    formatted_messages = prompt.invoke({
    "context": context,
    "question": question
    })
    print(formatted_messages)
    response = llm_client.groq_llm(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_MODEL,
        prompt=formatted_messages,
    )

    return {
        "answer": response,
        "context": context
    }
# print(run_chat("What is diabetes?"))

