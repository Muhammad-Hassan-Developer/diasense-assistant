from openai import AsyncOpenAI
class OpenAILLM:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """Simple, clean execution of the prompts provided by the chain."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    async def invoke_stream(self, system_prompt: str, user_prompt: str):
        response = await self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True # This is the magic key
    )
        async for chunk in response:
            content = chunk.choices[0].delta.content
        if content:
            yield content # Yield tokens as they arrive    