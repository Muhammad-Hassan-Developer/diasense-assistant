from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader, PyPDFLoader, TextLoader

class Loader:
    def __init__(self):
        pass  # Aap yahan attributes add kar sakte hain agar chahiye

    def load_from_dir(self, path: str, glob: str, loader_cls):
        try:
            loader = DirectoryLoader(path, glob=glob, loader_cls=loader_cls)
            documents = loader.load()  # Ye documents list return karega
            return documents
        except Exception as e:
            print(f"Error loading documents from directory {path}: {e}")
            return []

    def web_load(self, url: str):
        """Load text content from a web page URL."""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            return "\n".join(texts)
        except Exception as e:
            print(f"Error loading web content from {url}: {e}")
            return ""


