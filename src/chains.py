from src.llms import OpenAILLM
gpt=OpenAILLM()
from src.retrieval import sementic_retrival
sr=sementic_retrival(k=5)
from src.loader import Loader
loader=Loader()
system_prompt=loader.load_prompt("prompts/system.text")
human_prompt=loader.load_prompt("prompts/human.text")
# sr_docs=sr.invoke("What is diabetes?")
# print(sr_docs)
query="what is diabetes?"
docs = sr.invoke(query)
print(docs)
docs_text = "\n\n".join([doc.page_content for doc in docs])

final_human_prompt = human_prompt.format(
    context=docs_text,
    question=query
)
gpt_response=gpt.invoke(system_prompt=system_prompt,user_prompt=final_human_prompt)
print(gpt_response)
# import os
# import pickle
# from src.splitter import Splitter
# from src.loader import Loader
# from langchain_community.document_loaders import PyPDFLoader

# # Initialize loader and splitter
# loader = Loader()
# splitter = Splitter(chunk_size=700, chunk_overlap=100)

# print("Loading documents from PDFs...")

# # Load PDF documents
# docs_pdf = loader.load_from_dir(
#     "data/pdfs",
#     glob="**/*.pdf",
#     loader_cls=PyPDFLoader
# )

# print(f"Total documents loaded: {len(docs_pdf)}")
# print(type(docs_pdf))
# print(docs_pdf[0].page_content[:500])

# # Split documents into chunks
# print("Splitting documents into chunks...")
# chunks = splitter.split_documents(docs_pdf)

# print(f"Total chunks created: {len(chunks)}")

# # Create folder if it does not exist
# os.makedirs("data/processed", exist_ok=True)

# # Save chunks locally
# save_path = "data/processed/chunks_700_100.pkl"

# with open(save_path, "wb") as f:
#     pickle.dump(chunks, f)

# print(f"Chunks saved at {save_path}")
# from src.bm25_retrieval import BM25Manager
# bm=BM25Manager(chunk_path=save_path)
# bm.load_chunks()
# bm.build_retriever()
# # bm.query("What is diabetes?")
# print(bm.query("classification")[0].page_content)
