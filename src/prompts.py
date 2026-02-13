from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)

class Prompts:
    @staticmethod
    def retrieval_prompt(prompt: str):
        return PromptTemplate.from_template(prompt).template
    @staticmethod
    def human_prompt(self, prompt: str, chat_inputs: dict={}):
        return HumanMessagePromptTemplate.from_template(prompt).format(**chat_inputs)
    @staticmethod
    def system_prompt(self, prompt: str):
        return SystemMessagePromptTemplate.from_template(prompt)
    @staticmethod
    def diasense_chat_prompt(system_msg: str, human_msg: str):
        return ChatPromptTemplate.from_messages(
            [(SystemMessagePromptTemplate.from_template(system_msg)),
            (HumanMessagePromptTemplate.from_template(human_msg)),
            ]
        )
    
