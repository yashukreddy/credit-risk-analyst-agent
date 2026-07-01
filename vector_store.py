import os
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from embeddings import get_embeddings

load_dotenv()

def get_vector_store(namespace: str | None = None) -> PineconeVectorStore:
    """
    Connect to an existing Pinecone index using LangChain's PineconeVectorStore.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    namespace = namespace if namespace is not None else os.getenv("PINECONE_NAMESPACE")

    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY in .env")
    if not index_name:
        raise ValueError("Missing PINECONE_INDEX_NAME in .env")

    embeddings = get_embeddings()

    # Connect to an already-created Pinecone index
    # (Index must have the correct embedding dimension for the embedding model you use.)
    vs = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    return vs


if __name__ == "__main__":
    print("🧪 Testing Pinecone VectorStore connection...")

    try:
        vs = get_vector_store()
        # If your index is empty, this will just return [] (that's OK for now).
        results = vs.similarity_search("salary tax", k=2)
        print(f"✅ Connected. Retrieved {len(results)} docs.")
        if results:
            print("Sample result metadata:", results[0].metadata)
    except Exception as e:
        print("❌ Pinecone connection test failed:", e)
