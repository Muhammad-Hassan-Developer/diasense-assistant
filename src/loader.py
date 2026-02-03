from langchain_community.document_loaders import PyMuPDFLoader,TextLoader
from src.helper import Helper

helper = Helper()

class Loader:
    def __init__(self):
        self.all_texts = []

    def load_pdf_dir(self, dir_path: str, glob: str = "*.pdf", loader_cls=PyMuPDFLoader):
        texts = helper.Load_text_convert_directory(dir_path, loader_cls, glob)
        self.all_texts.extend(texts)
        return texts

    def load_text_dir(self, dir_path: str, glob: str = "*.text", loader_cls=TextLoader):
        texts = helper.Load_text_convert_directory(dir_path, loader_cls, glob)
        self.all_texts.extend(texts)
        return texts    


# loader = Loader()
# texts = loader.load_pdf_dir("data/pdfs")
# print(f"Total documents loaded: {len(texts)}")
# print(f"First document content preview: {texts[0][:500]}")  # Print first 500 characters of the first document
