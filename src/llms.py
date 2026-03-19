# src/llm.py

from openai import OpenAI
from src.config import Config

config = Config()
class OpenAILLM:

    def __init__(self):
        self.client = OpenAI(api_key=config.open_ai_api)
        self.model = config.open_ai_llm_model

    def invoke(self, system_prompt, user_prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content