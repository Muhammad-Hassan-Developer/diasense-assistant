from openai import OpenAI


class OpenAIEmbedding:

    def __init__(self, model:str, api_key:str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding