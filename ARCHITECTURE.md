# DiaSense Assistant - Architecture

## Project Overview
DiaSense AI is a medical assistant specializing in diabetes, using RAG (Retrieval-Augmented Generation) pipeline with the "Standards of Care 2026" guidelines.

## Directory Structure
```
├── configs/                    # Configuration files
├── data/                       # Data storage
│   ├── pdfs/                   # Raw PDF documents
│   ├── processed/              # Processed chunks & cleaned data
│   ├── texts/                  # Extracted text files
│   └── web/                    # Web-scraped data
├── src/                        # Source code
│   ├── api/                    # FastAPI application
│   ├── prompts/                # Prompt templates
│   ├── evaluator.py            # Ragas evaluation module
│   ├── chains.py               # Main RAG chain orchestration
│   ├── llms.py                 # LLM wrappers (OpenAI)
│   ├── embeddings.py           # Embedding models
│   ├── vector_store.py         # Vector database (ChromaDB)
│   ├── retrieval.py            # Retrieval manager
│   ├── sementic_retrieval.py   # Semantic/vector search
│   ├── bm25_retrieval.py       # BM25 keyword search
│   ├── reranker.py             # Document reranking
│   ├── runnables.py            # LangChain runnable chains
│   ├── pdf_cleaner.py          # PDF text cleaning
│   ├── splitter.py             # Text splitting utilities
│   ├── ingestion.py            # Data ingestion pipeline
│   ├── loader.py               # Document loaders
│   ├── config.py               # Configuration management
│   └── helper.py               # Utility functions
├── tests/                      # Test suite
└── requirements.txt            # Dependencies
```

## RAG Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│        Hybrid Retrieval         │
│  ┌─────────────┐  ┌──────────┐ │
│  │  Semantic   │  │   BM25   │ │
│  │  (ChromaDB) │  │ (Keyword)│ │
│  └──────┬──────┘  └────┬─────┘ │
│         └──────┬───────┘       │
│                ▼               │
│        Merge & Deduplicate     │
└────────────────┬────────────────┘
                 ▼
        ┌────────────────┐
        │  Format Context│
        └────────┬───────┘
                 ▼
        ┌────────────────┐
        │   LLM (GPT)    │
        │  Generation    │
        └────────┬───────┘
                 ▼
        ┌────────────────┐
        │   Evaluation   │ ← RagasEvaluator (optional)
        │ (Faithfulness, │
        │  Relevancy,    │
        │  Precision)    │
        └────────┬───────┘
                 ▼
           Final Response
```

## Components

### 1. Configuration (`config.py`)
- Centralized settings for API keys, model names, database configs

### 2. Embeddings (`embeddings.py`)
- OpenAI embedding model wrapper

### 3. Vector Store (`vector_store.py`)
- ChromaDB integration for vector storage
- Collection management

### 4. Retrieval
- **Semantic Retrieval** (`sementic_retrieval.py`): Vector similarity search
- **BM25 Retrieval** (`bm25_retrieval.py`): Keyword-based retrieval
- **Retrieval Manager** (`retrieval.py`): Orchestrates both retrieval methods
- **Reranker** (`reranker.py`): Re-ranks retrieved documents

### 5. LLM (`llms.py`)
- OpenAI AsyncOpenAI wrapper
- Supports both regular and streaming modes

### 6. Chains (`chains.py`)
- Main orchestration layer
- Hybrid retrieval chain (Semantic + BM25)
- Query chain with performance tracking
- Prompt templates for system & human messages

### 7. Evaluator (`evaluator.py`)
- **RagasEvaluator class**
- Metrics: Faithfulness, Answer Relevancy, Context Precision
- Can be called after query_chain() for quality assessment

## Data Flow

1. **Ingestion**: PDFs → Clean → Split → Embed → Store in ChromaDB
2. **Query**: User Question → Hybrid Retrieval → Merge → Format → LLM → Response
3. **Evaluation** (Optional): Query + Context + Response → RagasEvaluator → Scores

## Key Design Decisions
- Hybrid retrieval combines semantic understanding with keyword precision
- Async operations throughout for API performance
- LangChain Runnables for composable chains
- Ragas for automated evaluation quality metrics