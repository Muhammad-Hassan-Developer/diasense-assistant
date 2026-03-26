from __future__ import annotations
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel

from src.bm25_retrieval import BM25Manager
from src.loader import Loader
from src.llms import OpenAILLM
from src.retrieval import sementic_retrival

# Global Constants
DEFAULT_TOP_K = 5

def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _dedupe_and_format(docs: Sequence[Document], k: int) -> List[Document]:
    seen = set()
    unique_docs = []
    for d in docs:
        cleaned_content = _clean_text(d.page_content)
        if cleaned_content and cleaned_content not in seen:
            seen.add(cleaned_content)
            unique_docs.append(Document(page_content=cleaned_content, metadata=d.metadata))
        if len(unique_docs) >= k:
            break
    return unique_docs

@dataclass(frozen=True)
class HybridRAGConfig:
    top_k: int = DEFAULT_TOP_K
    bm25_path: str = "data/processed/chunks_700_100.pkl"
    sys_prompt_path: str = "prompts/system.txt"
    human_prompt_path: str = "prompts/human.txt"

class HybridRAG:
    def __init__(self, config: HybridRAGConfig):
        self.config = config
        self.loader = Loader()
        self.llm = OpenAILLM()
        self.semantic_retriever = sementic_retrival(k=config.top_k)
        
        self.bm25_manager = None
        if os.path.exists(config.bm25_path):
            self.bm25_manager = BM25Manager(chunk_path=config.bm25_path)
            self.bm25_manager.build_retriever(k=config.top_k)

        # Fallback handling for prompt extensions (.txt or .text)
        self.sys_prompt = self._load_p(config.sys_prompt_path)
        self.hum_prompt = self._load_p(config.human_prompt_path)
        self._chain = self._init_chain()

    def _load_p(self, path: str) -> str:
        try: return self.loader.load_prompt(path)
        except: return self.loader.load_prompt(path.replace(".txt", ".text"))

    def _init_chain(self):
        # Parallel Retrieval
        retrievers = {
            "semantic": RunnableLambda(lambda q: self.semantic_retriever.invoke(q)),
            "bm25": RunnableLambda(lambda q: self.bm25_manager.retriever.invoke(q) if self.bm25_manager else [])
        }
        
        def combine_step(data: Dict[str, Any]):
            query = data["question"]
            raw_docs = data["docs"]["semantic"] + data["docs"]["bm25"]
            final_docs = _dedupe_and_format(raw_docs, self.config.top_k)
            
            context = "\n\n".join([d.page_content for d in final_docs])
            prompt = self.hum_prompt.format(context=context, question=query)
            
            answer = self.llm.invoke(system_prompt=self.sys_prompt, user_prompt=prompt)
            return {"answer": answer, "documents": final_docs}

        return (
            RunnableLambda(lambda q: {"question": q, "docs": RunnableParallel(**retrievers).invoke(q)})
            | RunnableLambda(combine_step)
        )

    def run(self, query: str):
        return self._chain.invoke(query)

@lru_cache(maxsize=1)
def get_pipeline():
    return HybridRAG(HybridRAGConfig())

def query_chain(question: str):
    return get_pipeline().run(question)