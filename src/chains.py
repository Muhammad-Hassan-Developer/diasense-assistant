from __future__ import annotations

# NOTE: duplicate historical content below previously introduced extra
# `from __future__ import annotations` statements mid-file. Those are now removed/disabled.

from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional, Sequence

try:
    # LangChain v0.1+ (recommended)
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda, RunnableMap
except ModuleNotFoundError:  # pragma: no cover
    # Older LangChain versions (pre-split). Keep a soft fallback.
    from langchain.schema import Document  # type: ignore
    from langchain.schema.runnable import RunnableLambda, RunnableMap  # type: ignore

from src.bm25_retrieval import BM25Manager
from src.loader import Loader
from src.llms import OpenAILLM
from src.retrieval import sementic_retrival


DEFAULT_TOP_K = 5


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_document(doc: Document) -> Document:
    cleaned = _collapse_whitespace(doc.page_content)
    return Document(page_content=cleaned, metadata=dict(doc.metadata or {}))


def _dedupe_documents(docs: Sequence[Document]) -> List[Document]:
    seen: set[str] = set()
    out: List[Document] = []

    for d in docs:
        cleaned = _clean_document(d)
        key = cleaned.page_content
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)

    return out


def _round_robin_merge(a: Sequence[Document], b: Sequence[Document]) -> List[Document]:
    merged: List[Document] = []
    max_len = max(len(a), len(b))
    for i in range(max_len):
        if i < len(a):
            merged.append(a[i])
        if i < len(b):
            merged.append(b[i])
    return merged


def _docs_to_context(docs: Sequence[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs if d.page_content])


def _load_prompt_with_fallback(loader: Loader, relative_path: str) -> str:
    """
    Loads a prompt relative to `src/`.
    Prefers `.txt`, falls back to existing `.text` in this repo.
    """
    candidates = [relative_path]
    if relative_path.endswith(".txt"):
        candidates.append(relative_path[:-4] + ".text")
    elif relative_path.endswith(".text"):
        candidates.insert(0, relative_path[:-5] + ".txt")

    base_path = os.path.dirname(__file__)
    for rel in candidates:
        full_path = os.path.join(base_path, rel)
        if os.path.exists(full_path):
            return loader.load_prompt(rel)

    return loader.load_prompt(candidates[0])


@dataclass(frozen=True)
class HybridRAGConfig:
    top_k: int = DEFAULT_TOP_K
    bm25_chunk_path: Optional[str] = None
    system_prompt_path: str = "prompts/system.txt"
    human_prompt_path: str = "prompts/human.txt"


class HybridRAG:
    """
    Hybrid RAG pipeline using LangChain runnables (not deprecated chains):
    BM25 + semantic retrieval -> clean/dedupe -> prompt -> OpenAI LLM -> answer.
    """

    def __init__(self, config: HybridRAGConfig = HybridRAGConfig()):
        self.config = config
        self.loader = Loader()
        self.llm = OpenAILLM()

        # Semantic retriever (LangChain retriever: .invoke(question) -> List[Document])
        self.semantic_retriever = sementic_retrival(k=self.config.top_k)

        # BM25 retriever is optional; only enabled when chunk_path is provided.
        self.bm25_manager: Optional[BM25Manager] = None
        if self.config.bm25_chunk_path:
            self.bm25_manager = BM25Manager(chunk_path=self.config.bm25_chunk_path)
            self.bm25_manager.build_retriever(k=self.config.top_k)

        self.system_prompt = _load_prompt_with_fallback(self.loader, self.config.system_prompt_path)
        self.human_prompt = _load_prompt_with_fallback(self.loader, self.config.human_prompt_path)

        self._chain = self._build_chain()

    def _bm25_invoke(self, question: str) -> List[Document]:
        if not self.bm25_manager or not self.bm25_manager.retriever:
            return []
        return list(self.bm25_manager.retriever.invoke(question))

    def _semantic_invoke(self, question: str) -> List[Document]:
        return list(self.semantic_retriever.invoke(question))

    def _build_chain(self):
        bm25_runnable = RunnableLambda(lambda question: self._bm25_invoke(question))
        semantic_runnable = RunnableLambda(lambda question: self._semantic_invoke(question))

        retrieval_parallel = RunnableMap(bm25=bm25_runnable, semantic=semantic_runnable)

        def combine_and_clean(results: Dict[str, List[Document]]) -> List[Document]:
            bm25_docs = results.get("bm25", []) or []
            semantic_docs = results.get("semantic", []) or []
            merged = _round_robin_merge(semantic_docs, bm25_docs)
            deduped = _dedupe_documents(merged)
            return deduped[: self.config.top_k]

        retrieval_chain = (
            RunnableLambda(lambda x: x["question"] if isinstance(x, dict) else x)
            | retrieval_parallel
            | RunnableLambda(combine_and_clean)
        )

        def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
            question: str = payload["question"]
            docs: List[Document] = payload["documents"]
            context = _docs_to_context(docs)

            user_prompt = self.human_prompt.format(context=context, question=question)
            answer = self.llm.invoke(system_prompt=self.system_prompt, user_prompt=user_prompt)
            return {"answer": answer, "documents": docs}

        chain = (
            RunnableLambda(lambda q: {"question": q})
            | RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "documents": retrieval_chain.invoke({"question": x["question"]}),
                }
            )
            | RunnableLambda(generate)
        )

        return chain

    def invoke(self, question: str) -> Dict[str, Any]:
        return self._chain.invoke(question)


_DEFAULT_PIPELINE: Optional[HybridRAG] = None


def query_chain(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    bm25_chunk_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Required entry point:
      query_chain(question) -> {"answer": str, "documents": List[Document]}

    Hybrid search is enabled when `bm25_chunk_path` is provided.
    Prompts are loaded from `prompts/system.txt` + `prompts/human.txt` with fallback to `.text`.
    """
    global _DEFAULT_PIPELINE

    if (
        _DEFAULT_PIPELINE is None
        or _DEFAULT_PIPELINE.config.top_k != top_k
        or _DEFAULT_PIPELINE.config.bm25_chunk_path != bm25_chunk_path
    ):
        _DEFAULT_PIPELINE = HybridRAG(HybridRAGConfig(top_k=top_k, bm25_chunk_path=bm25_chunk_path))

    return _DEFAULT_PIPELINE.invoke(question)

# from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional, Sequence

try:
    # LangChain v0.1+ (recommended)
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda, RunnableMap
except ModuleNotFoundError:  # pragma: no cover
    # Older LangChain versions (pre-split). Keep a soft fallback.
    from langchain.schema import Document  # type: ignore
    from langchain.schema.runnable import RunnableLambda, RunnableMap  # type: ignore

from src.bm25_retrieval import BM25Manager
from src.loader import Loader
from src.llms import OpenAILLM
from src.retrieval import sementic_retrival


DEFAULT_TOP_K = 5


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_document(doc: Document) -> Document:
    cleaned = _collapse_whitespace(doc.page_content)
    return Document(page_content=cleaned, metadata=dict(doc.metadata or {}))


def _dedupe_documents(docs: Sequence[Document]) -> List[Document]:
    seen: set[str] = set()
    out: List[Document] = []

    for d in docs:
        cleaned = _clean_document(d)
        key = cleaned.page_content
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)

    return out


def _round_robin_merge(a: Sequence[Document], b: Sequence[Document]) -> List[Document]:
    merged: List[Document] = []
    max_len = max(len(a), len(b))
    for i in range(max_len):
        if i < len(a):
            merged.append(a[i])
        if i < len(b):
            merged.append(b[i])
    return merged


def _docs_to_context(docs: Sequence[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs if d.page_content])


def _load_prompt_with_fallback(loader: Loader, relative_path: str) -> str:
    """
    Loads a prompt relative to `src/`.
    Prefers `.txt`, falls back to existing `.text` in this repo.
    """
    candidates = [relative_path]
    if relative_path.endswith(".txt"):
        candidates.append(relative_path[:-4] + ".text")
    elif relative_path.endswith(".text"):
        candidates.insert(0, relative_path[:-5] + ".txt")

    base_path = os.path.dirname(__file__)
    for rel in candidates:
        full_path = os.path.join(base_path, rel)
        if os.path.exists(full_path):
            return loader.load_prompt(rel)

    return loader.load_prompt(candidates[0])


@dataclass(frozen=True)
class HybridRAGConfig:
    top_k: int = DEFAULT_TOP_K
    bm25_chunk_path: Optional[str] = None
    system_prompt_path: str = "prompts/system.txt"
    human_prompt_path: str = "prompts/human.txt"


class HybridRAG:
    """
    Hybrid RAG pipeline using LangChain runnables (not deprecated chains):
    BM25 + semantic retrieval -> clean/dedupe -> prompt -> OpenAI LLM -> answer.
    """

    def __init__(self, config: HybridRAGConfig = HybridRAGConfig()):
        self.config = config
        self.loader = Loader()
        self.llm = OpenAILLM()

        self.semantic_retriever = sementic_retrival(k=self.config.top_k)

        self.bm25_manager: Optional[BM25Manager] = None
        if self.config.bm25_chunk_path:
            self.bm25_manager = BM25Manager(chunk_path=self.config.bm25_chunk_path)
            self.bm25_manager.build_retriever(k=self.config.top_k)

        self.system_prompt = _load_prompt_with_fallback(self.loader, self.config.system_prompt_path)
        self.human_prompt = _load_prompt_with_fallback(self.loader, self.config.human_prompt_path)

        self._chain = self._build_chain()

    def _bm25_invoke(self, question: str) -> List[Document]:
        if not self.bm25_manager or not self.bm25_manager.retriever:
            return []
        return list(self.bm25_manager.retriever.invoke(question))

    def _semantic_invoke(self, question: str) -> List[Document]:
        return list(self.semantic_retriever.invoke(question))

    def _build_chain(self):
        bm25_runnable = RunnableLambda(lambda question: self._bm25_invoke(question))
        semantic_runnable = RunnableLambda(lambda question: self._semantic_invoke(question))

        retrieval_parallel = RunnableMap(bm25=bm25_runnable, semantic=semantic_runnable)

        def combine_and_clean(results: Dict[str, List[Document]]) -> List[Document]:
            bm25_docs = results.get("bm25", []) or []
            semantic_docs = results.get("semantic", []) or []
            merged = _round_robin_merge(semantic_docs, bm25_docs)
            deduped = _dedupe_documents(merged)
            return deduped[: self.config.top_k]

        retrieval_chain = (
            RunnableLambda(lambda x: x["question"] if isinstance(x, dict) else x)
            | retrieval_parallel
            | RunnableLambda(combine_and_clean)
        )

        def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
            question: str = payload["question"]
            docs: List[Document] = payload["documents"]
            context = _docs_to_context(docs)

            user_prompt = self.human_prompt.format(context=context, question=question)
            answer = self.llm.invoke(system_prompt=self.system_prompt, user_prompt=user_prompt)
            return {"answer": answer, "documents": docs}

        chain = (
            RunnableLambda(lambda q: {"question": q})
            | RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "documents": retrieval_chain.invoke({"question": x["question"]}),
                }
            )
            | RunnableLambda(generate)
        )

        return chain

    def invoke(self, question: str) -> Dict[str, Any]:
        return self._chain.invoke(question)


_DEFAULT_PIPELINE: Optional[HybridRAG] = None


def query_chain(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    bm25_chunk_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Required entry point:
      query_chain(question) -> {"answer": str, "documents": List[Document]}

    Hybrid search is enabled when `bm25_chunk_path` is provided.
    Prompts are loaded from `prompts/system.txt` + `prompts/human.txt` with fallback to `.text`.
    """
    global _DEFAULT_PIPELINE

    if (
        _DEFAULT_PIPELINE is None
        or _DEFAULT_PIPELINE.config.top_k != top_k
        or _DEFAULT_PIPELINE.config.bm25_chunk_path != bm25_chunk_path
    ):
        _DEFAULT_PIPELINE = HybridRAG(HybridRAGConfig(top_k=top_k, bm25_chunk_path=bm25_chunk_path))

    return _DEFAULT_PIPELINE.invoke(question)

# from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any, Dict, List, Optional, Sequence

try:
    # LangChain v0.1+ (recommended)
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda, RunnableMap
except ModuleNotFoundError:  # pragma: no cover
    # Older LangChain versions (pre-split). Keep a soft fallback.
    from langchain.schema import Document  # type: ignore
    from langchain.schema.runnable import RunnableLambda, RunnableMap  # type: ignore

from src.bm25_retrieval import BM25Manager
from src.loader import Loader
from src.llms import OpenAILLM
from src.retrieval import sementic_retrival


DEFAULT_TOP_K = 5


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _clean_document(doc: Document) -> Document:
    cleaned = _collapse_whitespace(doc.page_content)
    return Document(page_content=cleaned, metadata=dict(doc.metadata or {}))


def _dedupe_documents(docs: Sequence[Document]) -> List[Document]:
    seen: set[str] = set()
    out: List[Document] = []

    for d in docs:
        cleaned = _clean_document(d)
        key = cleaned.page_content
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)

    return out


def _round_robin_merge(a: Sequence[Document], b: Sequence[Document]) -> List[Document]:
    merged: List[Document] = []
    max_len = max(len(a), len(b))
    for i in range(max_len):
        if i < len(a):
            merged.append(a[i])
        if i < len(b):
            merged.append(b[i])
    return merged


def _docs_to_context(docs: Sequence[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs if d.page_content])


def _load_prompt_with_fallback(loader: Loader, relative_path: str) -> str:
    """
    Loads a prompt relative to `src/`.
    Prefers `.txt`, falls back to existing `.text` in this repo.
    """
    candidates = [relative_path]
    if relative_path.endswith(".txt"):
        candidates.append(relative_path[:-4] + ".text")
    elif relative_path.endswith(".text"):
        candidates.insert(0, relative_path[:-5] + ".txt")

    base_path = os.path.dirname(__file__)
    for rel in candidates:
        full_path = os.path.join(base_path, rel)
        if os.path.exists(full_path):
            return loader.load_prompt(rel)

    return loader.load_prompt(candidates[0])


@dataclass(frozen=True)
class HybridRAGConfig:
    top_k: int = DEFAULT_TOP_K
    bm25_chunk_path: Optional[str] = None
    system_prompt_path: str = "prompts/system.txt"
    human_prompt_path: str = "prompts/human.txt"


class HybridRAG:
    """
    Hybrid RAG pipeline using LangChain runnables (not deprecated chains):
    BM25 + semantic retrieval -> clean/dedupe -> prompt -> OpenAI LLM -> answer.
    """

    def __init__(self, config: HybridRAGConfig = HybridRAGConfig()):
        self.config = config
        self.loader = Loader()
        self.llm = OpenAILLM()

        # Semantic retriever (LangChain retriever: .invoke(question) -> List[Document])
        self.semantic_retriever = sementic_retrival(k=self.config.top_k)

        # BM25 retriever is optional; only enabled when chunk_path is provided.
        self.bm25_manager: Optional[BM25Manager] = None
        if self.config.bm25_chunk_path:
            self.bm25_manager = BM25Manager(chunk_path=self.config.bm25_chunk_path)
            self.bm25_manager.build_retriever(k=self.config.top_k)

        self.system_prompt = _load_prompt_with_fallback(self.loader, self.config.system_prompt_path)
        self.human_prompt = _load_prompt_with_fallback(self.loader, self.config.human_prompt_path)

        self._chain = self._build_chain()

    def _bm25_invoke(self, question: str) -> List[Document]:
        if not self.bm25_manager or not self.bm25_manager.retriever:
            return []
        return list(self.bm25_manager.retriever.invoke(question))

    def _semantic_invoke(self, question: str) -> List[Document]:
        return list(self.semantic_retriever.invoke(question))

    def _build_chain(self):
        bm25_runnable = RunnableLambda(lambda question: self._bm25_invoke(question))
        semantic_runnable = RunnableLambda(lambda question: self._semantic_invoke(question))

        retrieval_parallel = RunnableMap(bm25=bm25_runnable, semantic=semantic_runnable)

        def combine_and_clean(results: Dict[str, List[Document]]) -> List[Document]:
            bm25_docs = results.get("bm25", []) or []
            semantic_docs = results.get("semantic", []) or []
            merged = _round_robin_merge(semantic_docs, bm25_docs)
            deduped = _dedupe_documents(merged)
            return deduped[: self.config.top_k]

        retrieval_chain = (
            RunnableLambda(lambda x: x["question"] if isinstance(x, dict) else x)
            | retrieval_parallel
            | RunnableLambda(combine_and_clean)
        )

        def generate(payload: Dict[str, Any]) -> Dict[str, Any]:
            question: str = payload["question"]
            docs: List[Document] = payload["documents"]
            context = _docs_to_context(docs)

            user_prompt = self.human_prompt.format(context=context, question=question)
            answer = self.llm.invoke(system_prompt=self.system_prompt, user_prompt=user_prompt)
            return {"answer": answer, "documents": docs}

        chain = (
            RunnableLambda(lambda q: {"question": q})
            | RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "documents": retrieval_chain.invoke({"question": x["question"]}),
                }
            )
            | RunnableLambda(generate)
        )

        return chain

    def invoke(self, question: str) -> Dict[str, Any]:
        return self._chain.invoke(question)


_DEFAULT_PIPELINE: Optional[HybridRAG] = None


def query_chain(question: str, *, top_k: int = DEFAULT_TOP_K, bm25_chunk_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Entry point:
      query_chain(question) -> {"answer": ..., "documents": ...}

    Hybrid search is enabled when `bm25_chunk_path` is provided.
    Prompts are loaded from `prompts/system.txt` + `prompts/human.txt` with fallback to `.text`.
    """
    global _DEFAULT_PIPELINE

    if (
        _DEFAULT_PIPELINE is None
        or _DEFAULT_PIPELINE.config.top_k != top_k
        or _DEFAULT_PIPELINE.config.bm25_chunk_path != bm25_chunk_path
    ):
        _DEFAULT_PIPELINE = HybridRAG(HybridRAGConfig(top_k=top_k, bm25_chunk_path=bm25_chunk_path))

    return _DEFAULT_PIPELINE.invoke(question)

from src.llms import OpenAILLM
gpt=OpenAILLM()
from src.retrieval import sementic_retrival
sr=sementic_retrival(k=5)
from src.loader import Loader
loader=Loader()
system_prompt=loader.load_prompt("prompts/system.text")
human_prompt=loader.load_prompt("prompts/human.text")
# sr_docs=sr.invoke("What is diabetes?")
# print(sr_docs)
query="what is diabetes?"
docs = sr.invoke(query)
print(docs)
docs_text = "\n\n".join([doc.page_content for doc in docs])

final_human_prompt = human_prompt.format(
    context=docs_text,
    question=query
)
gpt_response=gpt.invoke(system_prompt=system_prompt,user_prompt=final_human_prompt)
print(gpt_response)
# import os
# import pickle
# from src.splitter import Splitter
# from src.loader import Loader
# from langchain_community.document_loaders import PyPDFLoader

# # Initialize loader and splitter
# loader = Loader()
# splitter = Splitter(chunk_size=700, chunk_overlap=100)

# print("Loading documents from PDFs...")

# # Load PDF documents
# docs_pdf = loader.load_from_dir(
#     "data/pdfs",
#     glob="**/*.pdf",
#     loader_cls=PyPDFLoader
# )

# print(f"Total documents loaded: {len(docs_pdf)}")
# print(type(docs_pdf))
# print(docs_pdf[0].page_content[:500])

# # Split documents into chunks
# print("Splitting documents into chunks...")
# chunks = splitter.split_documents(docs_pdf)

# print(f"Total chunks created: {len(chunks)}")

# # Create folder if it does not exist
# os.makedirs("data/processed", exist_ok=True)

# # Save chunks locally
# save_path = "data/processed/chunks_700_100.pkl"

# with open(save_path, "wb") as f:
#     pickle.dump(chunks, f)

# print(f"Chunks saved at {save_path}")
# from src.bm25_retrieval import BM25Manager
# bm=BM25Manager(chunk_path=save_path)
# bm.load_chunks()
# bm.build_retriever()
# # bm.query("What is diabetes?")
# print(bm.query("classification")[0].page_content)
