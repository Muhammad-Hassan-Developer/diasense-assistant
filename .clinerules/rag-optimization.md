# Coding Standards for RAG Optimization
Your goal is to refactor and optimize the existing RAG codebase for scalability and performance.

## 1. Modular Architecture (Structure)
- **Separation of Concerns:** Keep RAG logic separate from API routes. 
    - `services/rag_service.py`: For embedding, retrieval, and LLM logic.
    - `api/routes.py`: For FastAPI endpoints.
    - `core/config.py`: For environment variables and constants.
- **Class-Based Design:** Use Python Classes for the RAG pipeline (e.g., `class RAGManager`) to maintain state and reuse connections.

## 2. Performance Optimization
- **Asynchronous Operations:** Use `async/await` for file I/O, database queries, and LLM calls to prevent blocking.
- **Efficient Chunking:** Implement "Semantic Chunking" or "Parent Document Retrieval" if simple splitting is underperforming.
- **Vector DB indexing:** Ensure ChromaDB is not re-indexing the same files. Implement a hashing check (MD5) before processing an upload.

## 3. Code Quality & Standards
- **Pydantic Validation:** All API inputs/outputs must use Pydantic schemas.
- **Type Hinting:** Mandatory type hints for all function signatures.
- **Logging:** Replace `print()` statements with the `logging` module to track RAG steps (e.g., "Retrieving context...", "Generating answer...").
- **Error Handling:** Use custom exception handlers for "PDF corrupted", "Vector DB timeout", or "LLM API failure".

## 4. Specific Refactoring Task
- Scan the current code for "Hardcoded Values" and move them to a `.env` file.
- Look for "Large Functions" (>40 lines) and break them into smaller, testable helper functions.