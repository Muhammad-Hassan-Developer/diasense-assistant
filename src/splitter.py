from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Splitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split Documents into chunks while preserving metadata.
        Each chunk becomes a new Document.
        """
        chunked_docs: List[Document] = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)

            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["chunk_index"] = i

                chunked_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                )

        return chunked_docs
