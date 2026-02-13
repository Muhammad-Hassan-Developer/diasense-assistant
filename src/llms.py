# src/llms.py

from langchain_groq import ChatGroq

class Llms_client:
    def __init__(self):
        pass

    def groq_llm(self, api_key: str, model: str, prompt) -> str:
        llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_retries=2,
        )

        response = llm.invoke(prompt)   # ✅ NO list wrapping
        return response.content         # ✅ text only
