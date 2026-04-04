class SementicRetrieval: # Name CamelCase mein rakhein
    def __init__(self, vectorstore, k: int ):
        self.vectorstore = vectorstore
        self.k = k  # Ye line zaroori hai
        # Retriever ko initialize hi yahan kar dein taake ye memory mein 'Warm' rahay
        self.retriever = self._build_retriever()

    def _build_retriever(self):
        """Internal method jo retriever object banata hai"""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )

    # def invoke(self, query: str):
    #     """Direct search karne ke liye asaan method"""
    #     return self.retriever.invoke(query)