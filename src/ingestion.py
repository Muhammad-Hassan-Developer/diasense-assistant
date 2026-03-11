from langchain_community.document_loaders import PyPDFLoader,TextLoader,UnstructuredExcelLoader,UnstructuredWordDocumentLoader,PyMuPDFLoader
from src.loader import Loader
from src.helper import Helper
from src.config import Config
config=Config()
from src.splitter import Splitter


from src.vector_store import VectorStore
vs=VectorStore()
help = Helper()
loader = Loader()
splitter = Splitter(chunk_size=1000, chunk_overlap=200)
print("Loading documents from PDFs and Text files...")
docs_pdf= loader.load_from_dir("data/pdfs", glob="**/*.pdf", loader_cls=PyPDFLoader)
print(f"Total documents loaded: {len(docs_pdf)}")

print("Splitting ...")
print(type(docs_pdf))
print(docs_pdf[0].page_content[:500])

# Extract raw text strings from documents
chunks_pdf=splitter.split_documents(docs_pdf)
print(f"Total chunks created from PDFs: {len(chunks_pdf)}")
print(chunks_pdf[0])
print("Creating_vectors with the embedding and storing to chorma_db...")
from src.embeddings import OpenAIEmbedding
embedding_model = OpenAIEmbedding(model=config.open_ai_embedding_model, api_key=config.open_ai_api)

vectorstore = vs.get_vectorstore(api_key=config.chroma_api_key,collection_name=config.chroma_collection, tenant=config.chroma_tenant, database=config.chroma_db,embedding_model=embedding_model)
# BATCH_SIZE = 100
# for i in range(0, len(chunks_pdf), BATCH_SIZE):
#     batch = chunks_pdf[i:i+BATCH_SIZE]
#     vectorstore.add_documents(batch)
# print("Documents added to the vector store.")
ans=vectorstore.similarity_search("What is diabetes?")
print(ans[0].page_content)
# python -m src.ingestion