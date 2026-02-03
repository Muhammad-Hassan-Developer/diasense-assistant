from langchain_community.document_loaders import DirectoryLoader

class Helper:
    def Load_text_convert_directory(self, dir_path: str, loader_cls, glob: str):
        try:
            loader = DirectoryLoader(
                dir_path,
                glob=glob,
                loader_cls=loader_cls
            )
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
            return texts
        except Exception as e:
            print(f"Error loading from directory {dir_path}: {e}")
            return []
