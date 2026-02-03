from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader

class Ingestion:
    def __init__(self):
        self.all_texts = []

    def load_pdf(self, file_path: str):
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            self.all_texts.extend(texts)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")

    def load_text(self, file_path: str):
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            self.all_texts.extend(texts)
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")

    def load_web(self, url: str):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            self.all_texts.extend(texts)
        except Exception as e:
            print(f"Error loading web page {url}: {e}")

    def ingest_all(self, data_sources: dict):
        """
        data_sources example:
        {
            "pdfs": ["path/to/file1.pdf", "path/to/file2.pdf"],
            "texts": ["path/to/file1.txt", "path/to/file2.txt"],
            "web": ["https://example.com/article1", "https://example.com/article2"]
        }
        """
        self.all_texts = []  # Clear previous data

        for pdf_path in data_sources.get("pdfs", []):
            self.load_pdf(pdf_path)

        for text_path in data_sources.get("texts", []):
            self.load_text(text_path)

        for url in data_sources.get("web", []):
            self.load_web(url)

        return self.all_texts


if __name__ == "__main__":
    ingestion = Ingestion()
    data = {
        "pdfs": ["data/sample.pdf"],
        "texts": ["data/sample.txt"],
        "web": ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
    }
    all_data = ingestion.ingest_all(data)
    print(f"Total documents loaded: {len(all_data)}")
