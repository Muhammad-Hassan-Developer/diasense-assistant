from src.loader import Loader

if __name__ == "__main__":
    loader = Loader()
    texts = loader.load_pdf_dir("data/pdfs")
    texts1 = loader.load_text_dir("data/texts")
    print(f"Total documents loaded: {len(texts1)}")
    print(f"First document content preview: {texts1[0][:500]}") 
    print(f"First document content preview: {texts1[1][:500]}") # Print first 500 characters of the first document