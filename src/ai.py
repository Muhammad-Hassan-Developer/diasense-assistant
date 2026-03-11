from openai import OpenAI
from src.config import Config
config=Config()
client = OpenAI(api_key=config.open_ai_api)

response = client.responses.create(
    model="gpt-5.4",
    input="Write a short bedtime story about a unicorn."
)

print(response.output_text)
# python -m pip install openai
