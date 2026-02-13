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
retriever_docs = diasense_vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})


rm=RunnableManager()
all_prompts = loader.load_from_dir("prompts/",glob="**/*.text",loader_cls=TextLoader)
question_prompt = all_prompts[2].page_content
system_prompt = all_prompts[1].page_content
human_prompt = all_prompts[0].page_content
print("Question Prompt:\n", question_prompt)
print("Human Prompt:\n", human_prompt)
print("System Prompt:\n", system_prompt)   
final_human_prompt = prompts.human_prompt(human_prompt,{"question": question_prompt, "context": "context here"})
print("\n--- HUMAN PROMPT OUTPUT ---\n")
print(final_human_prompt)

retrieval=rm.retrieval_Prpmpt_runnable()
chain =retrieval | retriever_docs
docs = chain.invoke(question_prompt)
print("\n--- RETRIEVED DOCUMENTS ---\n")
for i, doc in enumerate(docs, 1):
    print(f"[Document {i}]\n{doc.page_content[:500]}\n---")
context = "\n\n".join(
    doc.page_content for doc in docs
)
print(context)
final_human_prompt = prompts.human_prompt(human_prompt,{"question": question_prompt, "context": context})
print("\n--- FINAL HUMAN PROMPT ---\n")
print(final_human_prompt)
final_system_prompt =  prompts.system_prompt(system_prompt)
final_prompt = prompts.diasense_chat_prompt(human_msg=final_human_prompt, system_msg=final_system_prompt)
print(final_prompt)
messages = final_prompt.format_messages(**{})
from src.llms import Llms_client
llm_client = Llms_client()
response = llm_client.groq_llm(
    api_key=config.GROQ_API_KEY,
    model=config.GROQ_MODEL,
    prompt=messages,
)

# # print("\n--- LLM RESPONSE ---\n")
print(response)