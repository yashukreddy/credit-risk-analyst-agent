from __future__ import annotations

from typing import Optional, Dict, Any, List
import json

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field

from vector_store import get_vector_store


def _format_docs(docs: List[Document]) -> str:
    """Compact formatting so the agent can cite evidence."""
    if not docs:
        return "No matching financial document chunks found."

    lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        filename = meta.get("filename", "unknown_file")
        s3_path = meta.get("s3_path", "")
        chunk = (d.page_content or "").strip()
        if len(chunk) > 1200:
            chunk = chunk[:1200] + "..."

        lines.append(
            f"[Chunk {i}] file={filename} s3={s3_path}\n{chunk}"
        )
    return "\n\n".join(lines)


def retrieve_customer_docs(
    customer_id: str,
    query: str,
    k: int = 5,
    extra_filter: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Retrieve top-k relevant chunks from Pinecone, filtered by customer_id.
    """
    if not customer_id:
        raise ValueError("customer_id is required")
    if not query:
        raise ValueError("query is required")

    vector_store = get_vector_store()

    # Pinecone metadata filter (applied server-side during vector search)
    base_filter: Dict[str, Any] = {"customer_id": customer_id}
    if extra_filter:
        base_filter.update(extra_filter)

    docs = vector_store.similarity_search(
        query=query,
        k=int(k),
        filter=base_filter,
    )
    return _format_docs(docs)

@tool
def rag_financial_docs(input_str: str) -> str:
    """
    Search documents. Input format:
    JSON: {"customer_id": "C001", "query": "salary"}
    OR comma-separated: C001, salary
    """
    customer_id = ""
    query = ""
    
    # 1. Try parsing as JSON
    try:
        data = json.loads(input_str)
        if isinstance(data, dict):
            customer_id = data.get("customer_id")
            query = data.get("query")
    except json.JSONDecodeError:
        pass

    # 2. Fallback: Parse comma-separated string
    if not customer_id:
        parts = [p.strip() for p in input_str.split(",", 1)]
        if len(parts) >= 2:
            customer_id = parts[0]
            query = parts[1].replace('"', '').replace("'", "")
        else:
            return "Error: Please provide BOTH customer_id and query (e.g., 'C001, salary slip')"

    return retrieve_customer_docs(customer_id, query, k=2)
    


if __name__ == "__main__":
    # Quick manual test (requires your index to have ingested docs with metadata.customer_id)
    print(
        rag_financial_docs.invoke(
            "C001, salary tax"
        )
    )
