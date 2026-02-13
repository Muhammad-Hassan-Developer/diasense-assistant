from langchain_core.runnables import RunnableLambda
from src.llms import Llms_client
from src.config import Config
from src.prompts import Prompts
llm_client = Llms_client()
config=Config()
prompts=Prompts()
class RunnableManager:
    def __init__(self):
        pass
    def groq_llm_runnable(self):
        def _call_llm(prompt):
            return llm_client.groq_llm(
                api_key=config.GROQ_API_KEY,
                model=config.GROQ_MODEL,
                prompt=prompt,
            )
        return RunnableLambda(_call_llm)

    # def retrieval_prompt_runnable(self):
    #     """Builds retrieval query text for Chroma"""
    #     def _call_retrieval(chat_inputs: dict):
    #         retrieval_msg = prompts.retrieval_prompt().format(**chat_inputs)
    #         return retrieval_msg.content
    #     return RunnableLambda(_call_retrieval)

    def retrieval_Prpmpt_runnable(self):
        def _call_retrieval(prompt: str):
            
            return prompts.retrieval_prompt(prompt)
        return RunnableLambda(_call_retrieval)
# rm=RunnableManager()
# test_prompt = "what is diabetes diagnosis criteria for a 45 year"
# retrieval=rm.retrieval_Prpmpt_runnable()
# retrieval_query=retrieval.invoke(test_prompt)
# print(retrieval_query)